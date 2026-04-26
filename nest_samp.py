import numpy as np
import matplotlib.pyplot as plt


true_mean = 5.0
data = np.random.normal(true_mean, 1.0, size=500)

def log_prior(mu):
    return -0.5 * ((mu - 3) / 3)**2

def log_likelihood(mu, data):
    return -0.5 * np.sum((data - mu)**2)

def log_posterior(mu, data):
    return log_prior(mu) + log_likelihood(mu, data)


def nested_sampling(data, n_live=50, n_iter=200):

    live_points = np.random.normal(0, 10, size=n_live)
    live_logL = np.array([log_posterior(x, data) for x in live_points])

    Z = 0.0
    X_prev = 1.0

    samples = []

    for _ in range(n_iter):

        worst_idx = np.argmin(live_logL)
        worst_point = live_points[worst_idx]
        worst_logL = live_logL[worst_idx]

        # shrink prior volume
        shrinkage = np.exp(-1.0 / n_live)
        X = X_prev * shrinkage

        weight = X_prev - X
        Z += np.exp(worst_logL) * weight

        samples.append(worst_point)

        # constrained replacement
        while True:
            proposal = np.random.normal(0, 10)
            logL = log_posterior(proposal, data)

            if logL > worst_logL:
                live_points[worst_idx] = proposal
                live_logL[worst_idx] = logL
                break

        X_prev = X

    return Z, np.array(samples)



# Run

Z, ns_samples = nested_sampling(data)
print("Estimated Evidence:", Z)
print("Estimated Mean:", np.mean(ns_samples))



plt.figure()
plt.hist(ns_samples, bins=40, density=True, alpha=0.7)
plt.axvline(true_mean, linestyle='--', color= "g", label='True Mean')
plt.axvline(np.mean(ns_samples), linestyle='--', color= "r",label='Nested Sampling Mean')
plt.title("Nested Sampling Approximation of Mean")
plt.legend()
plt.show()
