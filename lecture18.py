'''
lecture_18_gaussian_binom_approx_template.py
Gaussian distribution approximation to a Binomial distribution
CS251: Data Analysis and Visualization
Oliver W. Layton
Spring 2019
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

# n: Number of tosses
# p: Probability of getting water
# w: Sampling number of waters from 0 to n


n, p = 50, 0.5
w = np.arange(0, n+1)
plt.plot(w,stats.binom.pmf(w,n,p),label='Binomial')
plt.plot(w,stats.norm.pdf(w,n*p,np.sqrt(n*p*(1-p))),label='Gaussian')


plt.legend(loc=0, fontsize=13)

plt.title(f'n = {n}', fontsize=14)
plt.xlabel('Number of waters', fontsize=14)
plt.ylabel('Probability', fontsize=14)

plt.show()