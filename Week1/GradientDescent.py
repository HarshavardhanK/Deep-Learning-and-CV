# Gradient Descent implemented in Python with numpy module
import numpy as ny

def SquaredCostFunction(X, t, Y, size):
	f = ny.dot(X, t) # ignoring theta0 (yet)
	l = f - Y
	cost = ny.sum(l ** 2) / (2 * size) # / for the cost to be independant of the training set size 

	return cost

print(SquaredCostFunction(ny.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 4, ny.matrix([[8, 0, 0], [4, 9, 2], [6, 7, 8]]), 3))

def GradientDescent(X, Y, thet, a, size, times):
	X_trans = X.transpose()

	for i in range(0, times):

		f = ny.dot(X, thet) 
		loss = f - Y
		cost = ny.sum(loss ** 2) / (2 * size)
		
		g = ny.dot(X_trans, loss) / size
		thet = thet - a * g

		print(str(i) + " round, cost is " + str(cost))

	return thet


X = ny.matrix([[1], [4], [6]])
#print(X)
Y = ny.matrix([[8, 4, 9]])
#print(Y)

t = GradientDescent(X, Y, 3, 3.14, 3, 5)
print(t)


