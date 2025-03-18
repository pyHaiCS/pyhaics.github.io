# Experimental Models

This section presents a collection of benchmarking scenarios for comparing the performance and behavior of different Hamiltonian-based estimators based on their intrinsic properties:

1. **Correlation** in the Samples
2. **Reversibility** of the Chain
3. Influence of the **Importance Sampling** Re-Weighting (e.g., for MMHMC)

## Banana-Shaped Distribution

Given data $\{y_k\}_{k=1}^K$, we sample from the banana-shaped posterior distribution of the parameter $\theta = (\theta_1, \theta_2)$ for which the likelihood and prior distributions are respectively given as:

$$
\begin{aligned}
y_k|\theta &\sim \mathcal{N}(\theta_1+\theta_2^2, \sigma_y^2), \quad k=1, 2, \ldots, K\\
\theta_1,\theta_2 &\sim \mathcal{N}(0,\sigma_{\theta}^2)
\end{aligned}
$$

The sample data are generated with $\theta_1+\theta_2^2=1, \sigma_y=2, \sigma_{\theta}=1$. Then, the potential function is given by:

$$
U(\theta)=\dfrac{1}{2\sigma_y^2}\sum_{k=1}^K (y_k - \theta_1 - \theta_2^2)^2 + \log \left(\sigma_\theta^2\sigma_y^{100}\right)+\dfrac{1}{2\sigma_\theta^2}(\theta_1^2 + \theta_2^2)
$$

## Multivariate Gaussian Distribution

We sample from a $D$-dimensional Multivariate Gaussian Distribution $\mathcal{N} (0, \Sigma)$, where the precision matrix $\Sigma^{-1}$ is generated from a Wishart distribution.

For computational purposes, we take $D=1000$ dimensions and the covariance matrix to be diagonal with:

$$
\Sigma_{ii}=\sigma_i^2
$$

where $\sigma_i^2$ is the $i$-th smallest eigenvalue of the original covariance matrix. The potential function in this case is defined as:

$$
U(\theta)=\dfrac{1}{2}\theta^T \Sigma^{-1}\theta
$$

## Bayesian Logistic Regression (BLR)

Bayesian Logistic Regression (BLR) is the probabilistic extension of the traditional point-estimate logistic regression model by incorporating a prior distribution over the parameters of the model.

Given $K$ data instances $\{x_k, y_k\}_{k=1}^K$ where $x_k=(1, x_1, \ldots, x_D)$ are vectors of $D$ covariates and $y_k \in \{0, 1\}$ are the binary responses, the probability of a particular outcome is linked to the linear predictor function through the logit function:

$$
p(y_k| x_k, \theta) = \sigma(\theta^Tx_k) = \dfrac{1}{1+\exp(-\theta^Tx_k)}\\
\theta^Tx_k\equiv \operatorname{logit}(p_k) = \log \left(\dfrac{p_k}{1-p_k}\right)=\theta_0 + \theta_1 x_{1,k}+\ldots \theta_D x_{D, k}
```

where $\theta=(\theta_0, \theta_1, \ldots, \theta_D)^T$ are the parameters of the model, with $\theta_0$ denoted as the intercept.

The prior distribution over the parameters $\theta$ is chosen to be a Multivariate Gaussian distribution:

$$
\theta \sim \mathcal{N}(\mu, \Sigma), \quad \text{Usually } \theta \sim \mathcal{N}(0, I_{D+1})
```

where $\mu\in \mathbb{R}^{D+1}$ is the mean vector, $\Sigma \in \mathbb{R}^{D+1}$ is the covariance matrix, $0$ is the zero vector and $I_{D+1}$ is the identity matrix of order $D+1$.

### Available Datasets

The following datasets are included for benchmarking the BLR model:

| Dataset | D | K | Reference |
|---------|---|----|-----------|
| German | 25 | 1000 | [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Sonar | 61 | 208 | [Sonar Dataset](http://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) |
| Musk | 167 | 476 | [Musk Dataset (Version 1)](https://archive.ics.uci.edu/dataset/74/musk+version+1) |
| Secom | 444 | 1567 | [SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom) |

## Dynamic COVID-19 Epidemiological Models

A SEIR (Susceptible-Exposed-Infectious-Remove) dynamic compartmental mechanistic epidemiological model with a time-dependent transmission rate parametrized using Bayesian P-splines is applied to modeling the COVID-19 incidence data in the Basque Country (Spain).

The $\SEIR$ model consists of the following compartments:
- $S$: Number of individuals that are susceptible to be infected
- $E_1, \ldots, E_M$: Number of individuals at different stages of exposure (infected but not yet infectious)
- $I_1, \ldots, I_K$: Number of infectious individuals
- $R$: Number of individuals removed from the pool of susceptible individuals
- $C_I$: Counter of the total number of individuals that have been infected
- $\beta(t)$: Time-dependent transmission rate

The transmission rate $\beta(t)$ is modeled using B-splines:

$$
\log \beta(t) = \sum_{i=1}^m \beta_i B_i(t)
```

where $\{B_i (t)\}_{i=1}^m$ form a B-spline basis over the time interval $[t_0, t_1]$, with $m=q+d-1$ ($q$ is the number of knots, $d$ is the degree of the polynomials of the B-splines); and $\beta=(\beta_1, \ldots, \beta_m)$ is a vector of coefficients.

The $\SEIR$ model is governed by the following system of ODEs:

$$
\Dot{S}(t) = -\beta(t)S(t)\dfrac{I(t)}{N}\\
\Dot{E}_1 (t) = \exp \left(\sum_{i=1}^m \beta_i B_i(t)\right)S(t)\dfrac{I(t)}{N}-M\alpha E_1(t), \qquad \Dot{E}_M(t) = M\alpha E_{M-1}(t) - M\alpha E_M(t)\\
\Dot{I}_1 (t) = M\alpha E_M (t) - K\gamma I_1 (t), \qquad \Dot{I}_K (t) = K \gamma I_{K-1}(t) - K\gamma I_K (t)\\
\Dot{R}(t)=K\gamma I_K (t)\\
\Dot{C}_I (t) = \exp \left(\sum_{i=1}^m \beta_i B_i(t)\right)S(t)\dfrac{I(t)}{N}
```

with the following constraints:

$$
\begin{cases}
S(t_0) = N - E_0, E_1(t_0)=C_I (t_0) = E_0\\
E_2(t_0)=\cdots=E_M(t_0)=I_1(t_0)=\cdots=I_K(t_0)=R(t_0)=0\\
E(t) = \sum_{i=1}^M E_i(t)\\
I(t) = \sum_{j=1}^K I_j(t)\\
N = S(t) + E(t) + I(t) + R(t)
\end{cases}
$$

## Talbot Physical Effect

This benchmark analyzes Partial Differential Equations (PDEs) in the context of the phenomenon occurring when a plane light wave is diffracted by an infinite set of equally spaced slits (the grating, with distance $d$ between the slits).

We wish to find solutions to the following differential equation:

$$
\pdv[2]{u}{t}=\pdv[2]{u}{x}+\pdv[2]{u}{y}+\pdv[2]{u}{z}
```

in the domain $0 \leq x \leq \frac{d}{2}$, $z \geq 0$, $t \geq 0$ under the border conditions in $x$:

$$
\eval{\pdv{u}{x}}_{x=0}=\eval{\pdv{u}{x}}_{x=d/2}=0
```

the boundary conditions in $z$:

$$
u(t, x, z=0)=f(t,x)= \sin(\omega t)\theta(t)\, \chi\left(\dfrac{x}{w}\right)
```

and the initial conditions:

$$
u(t=0, x, z) = 0, \qquad \eval{\pdv{u}{t}}_{t=0} = 0
```

The solution can be expressed in closed-form as:

$$
u(t,x,z)=\sum_n g_n \left(\sin \omega (t-z) - k_n z \int_z^t \dfrac{J_1 (k_n \sqrt{\tau^2-z^2})}{\sqrt{\tau^2-z^2}}\sin \omega (t-\tau)\dd{\tau}\right)\theta (t-z) \cos k_n x
```

Solving the problem entails numerically approximating the complex integral, which involves:
1. A Bessel function of the first kind
2. An avoidable singularity as $\tau \rightarrow z$
3. A composition of two highly oscillatory functions