# Quick Start

In this introduction notebook we provide a simple implementation of a **Bayesian Logistic Regression (BLR)** model so that users can become familiarized with our library and how to implement their own computational statistics models.

First, we begin by importing `pyHaiCS`. Also, we can check the version running...
```python
import pyHaiCS as haics
print(f"Running pyHaiCS v.{haics.__version__}")
```

## Example - Bayesian Logistic Regression + HMC for Breast Cancer Classification
As a toy example, we implement below a classic BLR classifier for predicting (binary) breast cancer outcomes...
```python
import jax
import jax.numpy as jnp

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the data & convert to jax arrays
scaler = StandardScaler()
X_train = jnp.array(scaler.fit_transform(X_train))
X_test = jnp.array(scaler.transform(X_test))

# Add column of ones to the input data (for intercept terms)
X_train = jnp.hstack([X_train, jnp.ones((X_train.shape[0], 1))])
X_test = jnp.hstack([X_test, jnp.ones((X_test.shape[0], 1))])
```

First, we train a baseline point-estimate logistic regression model...
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def baseline_classifier(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(random_state = 42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy (w/ Scikit-Learn Baseline): {accuracy:.3f}\n")

baseline_classifier(X_train, y_train, X_test, y_test)
```

Now, in order to implement its Bayesian *counterpart*, we start by writing the model and the **Hamiltonian potential** (i.e., the negative log-posterior)...

```python
# Bayesian Logistic Regression model (in JAX)
@jax.jit
def model_fn(x, params):
    return jax.nn.sigmoid(jnp.matmul(x, params))

@jax.jit
def prior_fn(params):
    return jax.scipy.stats.norm.pdf(params)

@jax.jit
def log_prior_fn(params):
    return jnp.sum(jax.scipy.stats.norm.logpdf(params))

@jax.jit
def likelihood_fn(x, y, params):
    preds = model_fn(x, params)
    return jnp.prod(preds ** y * (1 - preds) ** (1 - y))

@jax.jit
def log_likelihood_fn(x, y, params):
    epsilon = 1e-7
    preds = model_fn(x, params)
    return jnp.sum(y * jnp.log(preds + epsilon) + (1 - y) * jnp.log(1 - preds + epsilon))

@jax.jit
def posterior_fn(x, y, params):
    return prior_fn(params) * likelihood_fn(x, y, params)

@jax.jit
def log_posterior_fn(x, y, params):
    return log_prior_fn(params) + log_likelihood_fn(x, y, params)

@jax.jit
def neg_posterior_fn(x, y, params):
    return -posterior_fn(x, y, params)

# Define a wrapper function to negate the log posterior
@jax.jit
def neg_log_posterior_fn(x, y, params):
    return -log_posterior_fn(x, y, params)
```

Then, we can call the `HMC` sampler in `pyHaiCS` (and sample from several chains at once) with a very *high-level* interface...
```python
# Initialize the model parameters (includes intercept term)
key = jax.random.PRNGKey(42)
mean_vector = jnp.zeros(X_train.shape[1])
cov_mat = jnp.eye(X_train.shape[1])
params = jax.random.multivariate_normal(key, mean_vector, cov_mat)

# HMC for posterior sampling
params_samples = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (X_train, y_train),
                            n_samples = 1000, burn_in = 200, 
                            step_size = 1e-3, n_steps = 100, 
                            potential = neg_log_posterior_fn,  
                            mass_matrix = jnp.eye(X_train.shape[1]), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120)

# Average across chains
params_samples = jnp.mean(params_samples, axis = 0)

# Make predictions using the samples
preds = jax.vmap(lambda params: model_fn(X_test, params))(params_samples)
mean_preds = jnp.mean(preds, axis = 0)
mean_preds = mean_preds > 0.5

# Evaluate the model
accuracy = jnp.mean(mean_preds == y_test)
print(f"Accuracy (w/ HMC Sampling): {accuracy}\n")
```