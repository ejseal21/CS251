'''
lecture16_linear_regression.py
Demo of using Numpy to compute linear regression, R^2, and p-value of slope test vs. slope = 0
Using diamond dataset
Oliver W. Layton
CS251: Data Analysis and Visualization
Spring 2019
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Load diamond data CSV file (included in seaborn module)
data = sns.load_dataset('diamonds')

# Randomly sample 500 points
data = data.sample(500)
print(data)
# Do the linear regression using
#   x: 'carat'
#   y: 'price'
x = data['depth']
y = data['price']

# Plot data sample
plt.scatter(x, y)

# superimpose linear regression line
m, b, r, p, stderr = stats.linregress(x, y)
plt.plot(x, m*x + b, 'r')

# Plot styling stuff
plt.title('Some points')
plt.xlabel('depth')
plt.ylabel('price')
plt.text(1, 15, f'$R^2$ = {r**2:.2f}', fontsize=15)
plt.text(2, 15, f'$p$ = {p:.2}', fontsize=15)

# Show us the plot in a pop-up window
plt.show()



def coeff(points):
	xsum = 0
	ysum = 0
	for point in points:
		xsum += point[0]
		ysum += point[1]
	xavg = xsum / len(points)
	yavg = ysum/ len(points)
	numerator = 0
	denominator = 0
	for point in points:
		numerator += (point[0] - xavg) * (point[1] - yavg)
		denominator += (point[0] - xavg) ** 2

	return numerator/denominator

points = [[1,1],[2,2],[4,2],[5,3]]
print(coeff(points))

def rsquared(points):
	yhat = coeff(points)
	ysum = 0
	for point in points:
		ysum += point[1]
	ybar = ysum/ len(points)
	
	numerator = 0
	denominator = 0
	for point in points:
		numerator += (point[1] - yhat) ** 2
		denominator += (point[1] - ybar) ** 2 
	return numerator/denominator

points = [[1,1],[2,2],[4,2],[5,3]]
print(rsquared(points))


xs = [0.955, 1.380, 1.854, 2.093, 2.674, 3.006, 3.255, 3.940, 4.060]
for i in range(len(xs)):
	xs[i] = xs[i] ** 2
print(sum(xs))