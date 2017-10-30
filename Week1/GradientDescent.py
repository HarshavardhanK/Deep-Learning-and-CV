# Gradient Descent implemented in Python with numpy module
import numpy as ny

def GradientDescent(X, Y, thet, a, size, times):
	X_trans = X.transpose()

	for i in range(0, times):

		f = ny.dot(X, thet) # ignoring theta0 (yet)
		loss = f - Y
		cost = ny.sum(loss ** 2) / (2 * size) # / for the cost to be independant of the training set size 
		
		g = ny.dot(X_trans, loss) / size
		thet = thet - a * g

		print(str(i) + " round, cost is " + str(cost))

	return thet


X = ny.matrix([[1, 2, 3], [1, 3, 4], [4, 5, 6]])
#print(X)
Y = ny.matrix([[4, 2, 8], [2, 3, 4], [7, 5, 6]])
#print(Y)

t = GradientDescent(X, Y, 3, 3.14, 3, 5)
print(t)


