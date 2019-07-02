import numpy as np
import scipy.stats as stats
import sys

def posterior(p_samps=40, waters=6,tosses=9,prior=None):
	p_grid = np.linspace(0,1,p_samps)

	if prior is None:
		prior = np.repeat(1, p_samps)

	binomial_probs = stats.binom.pmf(waters, tosses, p_grid)
	post = binomial_probs * prior
	post_norm = post / np.sum(post)