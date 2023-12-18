# BayesianInference4GW_GWData_Bootcamp

The complete materials for the `Gravitational Wave Data Exploration: A Practical Training in Programming and Analysis` can be found in this [GitHub repo](https://github.com/iphysresearch/GWData-Bootcamp).

## Talk: Bayesian inference for gravitational-wave science

### About the authors

- Seminar Coordinator: He Wang ( hewang@ucas.ac.cn )

- Seminar Designer: [Junjie Zhao](https://orcid.org/0000-0002-9233-3683) (junjiezhao@bnu.edu.cn)

Dr. [Junjie Zhao (赵俊杰)](https://orcid.org/0000-0002-9233-3683) received his Ph.D. degree in theoretical physics from Peking University in 2021 and is currently doing scientific research as a "LiYun" postdoctoral fellow (励耘博士后) in the Department of Astronomy, Beijing Normal University. The main research interests are gravitational-wave physics, testing gravity, physics of pulsar, etc.

### Table of Contents (内容概览)

Below is the overview of this seminar.

* Brief introduction to gravitational wave (引力波简要介绍)
- Part I: Bayesian inference (贝叶斯推断)
	- Definition of “probability” ("概率"的定义)
	- Rethink the interpretations (重思概率诠释)
		- Frequentist statistics (频率学派)
		- Bayesian statistics (贝叶斯学派)
	- Bayes' theorem (贝叶斯定理)
		- Application to the detection of gravitational wave (在引力波探测上应用)
	- Bayesian inference framework (贝叶斯推断框架)
		- Parameter estimation for gravitational-wave data (引力波数据分析中参数估计)
		- Model selection for gravitational-wave data (引力波数据分析中模型选择)
- Q & A

- Part II: Bayesian computation (贝叶斯计算方法)
	- Markov Chain Monte Carlo (MCMC; 马尔可夫链-蒙特卡罗方法)
		- hands-on tiny mcmc example
	- Nested sampling (嵌套采样)
		- hands-on tiny nested-sampling example
- Part III: All in gravitational-wave data (一切尽在引力波数据中)
	- Use Bilby & Parallel Bilby in the GW data analysis
	- Show the complete pipeline for the data analysis
- The AMAZING Thomas Bayes (为美好的世界献上"贝叶斯定理")
- Q & A


## The environment for the GW analysis

Here, I **recommend** using the `conda` or `mamba` command to manage your environment.
Please make sure you have installed the
[`Miniforge`](https://github.com/conda-forge/miniforge) /
[`Miniconda`](https://docs.conda.io/projects/miniconda/en/latest/) /
[`Anaconda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
software.

> Recommend: you can always replace the `conda` with the `mamba` (alternative faster conda) to
manage your environment.

You can run
```bash
conda --version
mamba --version
```
to check your `conda` or `mamba` environment.

> Warning: if you are using the `Windows` computer, please install
> [WSL2](https://learn.microsoft.com/en-us/windows/wsl/about) or use the remote
> Linux server to obtain the best experience.


### Create the `full` environment [Recommend for the professional user]

To create the `full` environment, run the following command:

```bash
bash ./envs/update_igwn_envs.sh
```

If your network is blocked, please try
```bash
conda env update -f ./envs/igwn-py310-linux-64.yaml
```
for the `Linux x86-64` and `Linux amd64` architecture.

For the `macOS x86-64` architecture
```bash
conda env update -f ./envs/igwn-py310-osx-64.yaml
```

For the `macOS arm64` (Apple silicon)
```bash
conda env update -f ./envs/igwn-py310-osx-arm64.yaml
```

### Create the `tiny` environment
```bash
conda create -n igwn-py310 python=3.10 numpy scipy lalsuite pycbc bilby parallel-bilby dynesty emcee jupyterlab ipympl ipywidgets
```

More advanced commands for `conda` / `mamba` can be found at [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)