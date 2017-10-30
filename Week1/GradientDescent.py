# Gradient Descent implemented in Python with numpy module
import numpy as ny

def SquaredCostFunction(X, t, Y, size):
	f = ny.dot(X, t) # ignoring theta0 (yet)
	l = f - Y
	cost = ny.sum(l ** 2) / (2 * size) # / for the cost to be independant of the training set size 

	return cost

print(SquaredCostFunction(ny.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 4, ny.matrix([[8, 0, 0], [4, 9, 2], [6, 7, 8]]), 3))

def GradientDescent(X, Y, c, m, a, size, times):
	X_trans = X.transpose()

	for i in range(0, times):

		f = c + ny.dot(X, m) # taking f(X) := c + mX
		loss = f - Y
		print(loss)
		cost = ny.sum(loss ** 2) / (2 * size)

		# take partial derivative manually..

		g0 = loss / size 
		g1 = ny.dot(X_trans, loss) / size

		#update the theta values

		c = c - a * g0
		m = m - a * g1

		print(str(i) + " round, cost is " + str(cost))

	return (c, m)


X = ny.matrix([[1], [4], [6]])
#print(X)
Y = ny.matrix([[6, 6, 7], [8, 6, 7], [3, 4 ,9]])
#print(Y)

t = GradientDescent(X, Y, 2, 3, 3.14, 3, 50)
print(t)


