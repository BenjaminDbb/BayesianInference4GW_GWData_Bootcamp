import numpy as np

def proposal(theta_prev, step_theta, theta_low_bound, theta_high_bound):
 
    # random walk
    theta_prop = np.random.normal(theta_prev, step_theta)
    
    # reflect proposals outside of priors' bounds
    idx_over_high = theta_prop > theta_high_bound
    idx_over_low = theta_prop < theta_low_bound

    theta_prop[idx_over_high] = 2*theta_high_bound[idx_over_high] - theta_prop[idx_over_high]
    theta_prop[idx_over_low] = 2*theta_low_bound[idx_over_low] - theta_prop[idx_over_low]
    
    return theta_prop

def log_posterior(theta, x, y, yerr):
    return log_likelihood(theta, x, y, yerr)

def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def main():

    # set the random number generator seed
    np.random.seed(123)

    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += np.abs(f_true * y) * np.random.randn(N)
    y += yerr * np.random.randn(N)
 
    # Bounds on Priors of m, b, log_f
    m_bounds        = np.array([-5.0, 0.5])
    b_bounds        = np.array([0.0,  10 ])
    log_f_bounds    = np.array([-10., 1.0])
    
    theta_bounds = np.array([m_bounds, b_bounds, log_f_bounds])
    theta_low_bound = theta_bounds[:,0]
    theta_high_bound = theta_bounds[:,1]
        
    # Carry out MCMC fitting to get best-fit parameters
    # set the MCMC parameter
    
    proposal_step_factor = 0.03
    f_burnin = 0.2
    N = 10000
    
    Nburnin = int(N * f_burnin)
    
    N_params = theta_bounds.shape[0]
    step_theta = proposal_step_factor * (theta_high_bound - theta_low_bound)
    
    theta = np.zeros((N, N_params))
    
    theta_prev = np.random.uniform(low=theta_low_bound, high=theta_high_bound)
    
    for i in range(N):
        # take random step using the proposal distribution
        theta_prop = proposal(theta_prev, step_theta, theta_low_bound, theta_high_bound)
        
        logP_prop = log_posterior(theta_prop, x, y, yerr)
        logP_prev = log_posterior(theta_prev, x, y, yerr)
        
        U = np.random.uniform(0.0, 1.0)
        r = np.min([1.0, np.exp(logP_prop - logP_prev)])
        
        if (r >= U):
            theta[i,:] = theta_prop
            theta_prev = theta_prop
        else:
            theta[i,:] = theta_prev

        # plot the result
        if (i % 100) == 0:
            print(theta[i,:])
            
    # burnin
    theta = theta[Nburnin:,:]

if __name__== "__main__":
    main()
