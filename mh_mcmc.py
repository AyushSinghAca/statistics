import numpy as np
import matplotlib.pyplot as plt


# As we dont have real data so generating 50 random values from gaussian distribution with mean 5 and std 1
true_mean = 5.0
data = np.random.normal(true_mean, 1.0, size=50)  


# we assume prior is gaussian 
def log_prior(mu):
    return -0.5 * ((mu - 2) / 5)**2  
    # prior belief: mu is around 2 but can vary a lot as std is 5


# how good mu explains the data
def log_likelihood(mu, data):
    return -0.5 * np.sum((data - mu)**2)
    # if mu is close to data values then likelihood is high


def log_posterior(mu, data):
    return log_prior(mu) + log_likelihood(mu, data)
    # combining belief + data evidence


def metropolis_hastings(data, n_samples=5000, proposal_std=0.2):
    samples = []
    mu_current = 1  # starting guess of mu

    for _ in range(n_samples):
        # propose new mu near current mu (random step) - This is core of MC
        mu_proposed = np.random.normal(mu_current, proposal_std)

        # check if new mu is better or not
        log_accept_ratio = (
            log_posterior(mu_proposed, data)
            - log_posterior(mu_current, data)
        )

        # accept if better, sometimes even if worse so used random  (important for exploration)
        if np.log(np.random.rand()) < log_accept_ratio:
            mu_current = mu_proposed

        # store current value of mu
        samples.append(mu_current)

    return np.array(samples)


# run MCMC
samples = metropolis_hastings(data)

# remove early bad guesses (burn-in)
burn_in = 1000
samples = samples[burn_in:]


print("Estimated mean:", np.mean(samples))
plt.hist(samples, bins=50, density=True)
plt.axvline(true_mean, color='g', label='True Mean')  
plt.axvline(np.mean(samples), color='r', linestyle='--', label='Sample Mean')  
plt.title("Posterior Distribution of Mean")
plt.legend()
plt.show()
