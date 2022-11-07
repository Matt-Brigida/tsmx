import jax.numpy as jnp
from jax import random
key = random.PRNGKey(0)

### How do we want to handle time here and step size???

def sim(alpha, beta, sigma, steps, years, initial):
    """
    Simulate the Vasicek short rate model:
    dr = alpha(beta - r)dt + sigma dW
    """
    delta_t = 1 / steps
    i = 0
    r = jnp.zeros(steps * years)
    r = r.at[0].set(initial)
    while (i < steps):
        r = r.at[i + 1].set(r.at[i] + alpha * (beta - r.at[i]) * (delta_t) + sigma * jnp.sqrt(delta_t) * random.normal(key, shape=(1,1)) # <- returns same value change



        i =+ 1
