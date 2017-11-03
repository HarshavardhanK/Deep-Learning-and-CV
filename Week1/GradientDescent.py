# Gradient Descent implemented in Python with numpy module
import numpy as ny
import matplotlib.pyplot as plt

def SquaredCostFunction(X, t, Y, size):
	f = ny.dot(X, t) # ignoring theta0 (yet)
	l = f - Y
	cost = ny.sum(l ** 2) / (2 * size) # / for the cost to be independant of the training set size

	return cost

print(SquaredCostFunction(ny.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 4, ny.matrix([[8, 0, 0], [4, 9, 2], [6, 7, 8]]), 3))

def GradientDescent(X, Y, a, times):

	X_trans = X.transpose()
	size = X.shape[0]
	m = ny.ones(2)

	for i in range(0, times):

		f = ny.dot(X, m) # taking f(X) := c + mX
		loss = f - Y
		#print(loss)
		cost = ny.sum(loss ** 2) / (2 * size)

		# take partial derivative manually..

	#	g0 = loss / size
		g1 = ny.dot(X_trans, loss) / size

		#update the m values
		m = m - a * g1

		print(str(i + 1) + " iter, cost is " + str(cost))

	return m


'''X = ny.matrix([[1], [4], [6]])
print(ny.shape(X))
Y = [6, 6, 7]
print(ny.shape(Y))

(c, m) = GradientDescent(X, Y, 2, 3, 3.14, 3, 50) # deprecated function

H = c + m * X
print(H)'''

path="/Users/HarshavardhanK/Downloads/machine-learning-ex1/ex1/ex1data1.txt"
data_file=open(path, 'r')
x_arr = []
y_arr = []

with open(path, "r") as f:
    for l in f:

        currentL = l.split(",")
        x_arr.append(float(currentL[0]))
        y_arr.append(float(currentL[1]))

'''for i in range(0, 3):
    x_arr.append(0)
    y_arr.append(0)'''

X = ny.array(x_arr)
X = X.reshape(97, 1)
y_arr = ny.array(y_arr)
#y_arr = y_arr.reshape(20,5)
(m, n) = ny.shape(X)
print((m, n))

X = ny.c_[ ny.ones(m), X] # insert column
a = 0.02 # learning rate
theta = GradientDescent(X, y_arr,a, 10000)
print(theta)
