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
    
    # Carry out Nested Sampling to get best-fit parameters	
    step = 0.01 # Initialguess suitable step-size in(0,1)
    N_mcmc = 15
    N = 600 # MCMC counter (pre-judged # steps)   
    nlive = 40 # The number of live points
    N_params = theta_bounds.shape[0]

    # init. the method
    
    theta_live = np.zeros((nlive, N_params))
    logL_live = np.zeros((nlive))
    
    theta = np.zeros((N, N_params))
    logL = np.zeros((N))
    
    dlogz = np.inf
    last_logZ = -np.inf
    step_theta = step * (theta_high_bound - theta_low_bound)
    
    # `Algorithm 1: Static Nested Sampling` from arXiv: 1904.02180
    # 9.3.2 Programming the lighthouse problem in ‘C’ from 
 
    # Initialize live points.
 
    # (1) draw `nlive` “live” points from the priors, 
     # 	  and calculate the log-likelihood vaules
    for i in range(nlive):
        theta_live[i,:] = np.random.uniform(theta_low_bound, theta_high_bound)
        logL_live[i] = log_likelihood(theta_live[i,:], x, y, yerr)
    
    # Nested-sampling begin
    # `Main sampling loop.`
    for i in range(N):
        
        # (2) sort the likelihoods and store the smallest one
        sorted_idx = np.argsort(logL_live)
        theta_live = theta_live[sorted_idx,:] # sorted `theta_live` list
        logL_live     = logL_live[sorted_idx] # sorted `logL_live` list
        
        # use the min-likelihood point: Lstar: logL[i]
        theta[i,:] = theta_live[0,:]
        logL[i] = logL_live[0]
        
        # (3) replace the point with a higher likelihood sampled point, 
        #     using Metropolis-Hastings MCMC (take N_mcmc steps)
          #     4.2.2 Random walks from arXiv: 1904.02180
        # the rand index for `theta_new` except `rnd_index = 0`
        rnd_index = np.random.randint(low=1, high=nlive)
        theta_new = theta_live[rnd_index,:]
  
        # Total number of accepted points with logL>logL*
        n_accept = 0
  
        # Total number of points proposed to within the ellipsoid and cube
        # but rejected due to logL<=logL* condition
        n_reject = 0
        
          # loop until n_accept = N_mcmc
        while (n_accept < N_mcmc):
      
            logL_new = log_likelihood(theta_new, x, y, yerr)
   
            # random walk for the next point
            theta_prop = proposal(theta_new, step_theta, theta_low_bound, theta_high_bound)
            logL_prop = log_likelihood(theta_prop, x, y, yerr)
            
            logP_prop = log_posterior(theta_prop, x, y, yerr)
            logP_new = log_posterior(theta_new, x, y, yerr)

            # MH sampling (actually, you can remove and just judge `logL_prop > logL[i]`)
            U = np.random.uniform(0.0, 1.0)
            r = np.min([1.0, np.exp(logP_prop - logP_new)])
   
            if (logL_prop > logL[i]) and (r >= U):
                n_accept += 1
                # update the parameter space
                theta_new = theta_prop
                logL_new = logL_prop
            else:
                n_reject += 1
    
        # refresh the `nlive` points
        theta_live[0] = theta_new
        logL_live[0] = logL_new

        # update the (adaptive mcmc) step sizes
        if (n_accept > n_reject):
            step *= np.exp(1.0/n_accept)
        else:
            step *= np.exp(-1.0/n_reject)

        step_theta = step * (theta_high_bound - theta_low_bound)

        # calculate evidence weights W and Bayesian evidence Z
          # W = logX = - np.arange(1,i+2) / nlive
        W = np.exp(-np.arange(1, i+2)/nlive)
        
        # find the maxium likelihood value
        logLmax = np.max(logL[:i+1])
    
        # compute Bayesian evidence Z
          # Z = \int_{0}^{1} L dX <= \sum_{i=1}^{m} (X_{i-1} - X_{i}) L_{i} + X_{m} L_{max}
        logZ_upper = np.log( np.sum( np.exp(logL[:i+1] - logLmax) * W) ) + logLmax
  
        dlogz = logZ_upper - last_logZ
        # Z = np.exp(logZ)
        
        last_logZ = logZ_upper 
        
        # plot proposed function
        if (i % 10) == 0:
            print(f"progress i={i}\tlogZ={logZ_upper:.6f}\tdlogz={dlogz:.6f}\tn_accept={n_accept}\tn_reject={n_reject}")
            # print("    logZ=", logZ_upper)
            # print("    dlogz=", dlogz)
            # print("    n_accept=", n_accept, " n_reject=", n_reject)
            # print("    theta=", theta[i,:])
            # print("    step=", step)


    # Resample results with proper weights to obtain posteriors
    N_resample = 10000
    theta_posterior = np.zeros((N_resample, N_params))
    
    W_sample = np.exp(logL + np.log(W) - logZ_upper)
    W_sample /= np.sum(W_sample)
    resample_idx = np.random.choice(range(N), N_resample, p=W_sample)
 
    for i in range(N_resample):
        theta_posterior[i,:] = theta[resample_idx[i],:]

if __name__== "__main__":
  main()

