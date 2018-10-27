# Excercise 1 Week 3 University or whatever acceptance classifier


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression


path='/Users/HarshavardhanK/Google Drive/Python/Project Manas/AndrewNG/machine-learning-ex2/ex2/ex2data1.txt'

data_frame = pd.read_csv(path, header=None)

size = len(data_frame[0])

exam1_arr = np.array(data_frame[0]).reshape(size, 1)
exam2_arr = np.array(data_frame[1]).reshape(size, 1)
result = np.array(data_frame[2]).reshape(size, 1)

def preprocess_data():

    new_exam1_arr = []
    new_exam2_arr = []
    new_result_arr = []


    for i in range(0, size):

        if result[i] == 0:

            new_exam1_arr.append(exam1_arr[i])
            new_exam2_arr.append(exam2_arr[i])
            new_result_arr.append(0)


    for i in range(0, size):

        if result[i] == 1:

            new_exam1_arr.append(exam1_arr[i])
            new_exam2_arr.append(exam2_arr[i])
            new_result_arr.append(1)



    return np.array(new_exam1_arr), np.array(new_exam2_arr), np.array(new_result_arr)


new_exam1_arr,  new_exam2_arr , y = preprocess_data()

X = np.dstack((new_exam1_arr, new_exam2_arr)).reshape(100, 2)

def find_accept_point():

    for i in range(0, size):
        if y[i] == 1:
            return i


point = find_accept_point()

plt.scatter(new_exam1_arr[0:point], new_exam2_arr[0:40], color='red', marker='x', label='Rejected')
plt.scatter(new_exam1_arr[point:size], new_exam2_arr[40:size], color='blue', marker='o', label='Accepted')
<<<<<<< HEAD
plt.xlabel("Exam 1 Marks")
plt.ylabel("Exam 2 Marks")
=======
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
>>>>>>> 6a18a5c2e778f58e0e57ac60a4533996c34c2b04
plt.show()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X, y)
plt.title("Acceptance Scatter Plot")
<<<<<<< HEAD
plt.xlabel("Exam 1 Marks")
plt.ylabel("Exam 2 Marks")
=======
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
>>>>>>> 6a18a5c2e778f58e0e57ac60a4533996c34c2b04
plot_decision_regions(X, y, clf=lr, res=0.2, legend=2)
plt.show()

print(lr.predict_proba(X[0, :]))
