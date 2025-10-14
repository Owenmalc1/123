""""

This code acts as demonstration of numpy array operations such as; creation and manipulation. It includes tasks to create arrays of zeros, ranges, and random numbers.

Tasks:
1. Create a 1D array of zeros with a specified length.
2. Create a 1D array with a range of values.
3. Create a 2D array of random numbers.

Argumens:
- m: Number of rows for the 2D array. (must be an integer)
- n: Number of columns for the 2D array. (must be an integer)

Returns:
- x: A 2D array of shape (m, 1) filled with zeros
- y: A 1D array with values from 1 to n
- z: A 2D array of shape (m, n) filled with random numbers from a uniform distribution over [0, 1)

"""


import numpy as np
from pexpect import which

# Task 1a

m = 5  # Set m to the desired number of rows
x = np.zeros((m,1))

n = 5
y = np.arange(1,n+1)

# 2D array of random numbers from [0,1) of shape (m,n)
z = np.random.rand(m,n)
print(z)

b = np.random.uniform(0, 1, (m, n))
print(b)

#Task 1b write a docstring for 1a

#Task 1c

def task1c_(y,z):
    mean_y = np.mean(y)
    max_z = np.max(z)
    print("Mean of y:", mean_y)
    print("Max of z:", max_z)

task1c_(y,z)

a = np.array([1, 2, 3, 4 ,5 ,6 ,7 ,8, 9, 10])
x = a**2

# Task 1d
def task1d_(a,x):
    print("Original array:", a)
    print("Squared array:", x)

task1d_(a,x)

# Explanation of mutable vs immutable objects:
# Mutable objects can be changed after they are created. For example, lists and NumPy arrays are mutable:
# you can modify their contents, add or remove elements, etc.
# Immutable objects cannot be changed after creation. For example, integers, floats, and strings are immutable:
# any operation that seems to change them actually creates a new object.

#4.2 square 2D array M of shape n×n 

def task2a_(n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = i + 2*j
    return M

M = task2a_(3)

print("Square 2D array M of shape (n, n):", M)

#Task 2b 1D array summing M_ij from j=0 to j = n-1 for each i

M = task2a_(3)

def task2b_(M):
    return np.sum(M, axis=1)

print("Sum of each row in M:", task2b_(M))

#4.3 write a function 'task3' that computes a gaussian log-likelihood from three vectors which are 1D arrays with 3 elements each

d = np.array([1.0, 2.0, 3.0])
mu = np.array([1.5, 2.5, 3.5])
sigma = np.array([0.1, 0.2, 0.3])

def task3_(d, mu, sigma):
    LogL = -0.5 * np.sum((d - mu) / sigma) ** 2
    return LogL
print("Gaussian log-likelihood:", task3_(d, mu, sigma))

task3_(d, mu, sigma)

# 4.4 write a function 'task4' that does basic data analysis of the data from a file 'data(2).txt' label plots correctly,
# save them to files
# write some code to save a nicely formatted text table of the numerical values that you find.
# taking the mean of each column and the standard deviation of each column
# make a histogram of each column
# make a scatter plot of column 1 against column 2
# make a scatter plot of column 1 against column 3
# make a scatter plot of column 2 against column 3
# save all plots to files

import matplotlib.pyplot as plt
data = np.loadtxt('data(2).txt')
omegamh2 = data[:,0]
omegabh2 = data[:,1]
H0 = data[:,2]
print(data)

def task4_(data):
    mean_omegamh2 = np.mean(data[:, 0])
    std_omegamh2 = np.std(data[:, 0])
    omegabh2 = np.mean(data[0:,1])
    std_omegabh2 = np.std(data[:, 1])
    mean_H0 = np.mean(data[:, 2])
    std_H0 = np.std(data[:, 2])
    print(f"Mean omegamh2: {mean_omegamh2}, Std: {std_omegamh2}")
    print(f"Mean omegabh2: {omegabh2}, Std: {std_omegabh2}")
    print(f"Mean H0: {mean_H0}, Std: {std_H0}")

    plt.hist(data[:, 0], bins=20, label='omegamh2')
    plt.xlabel('omegamh2')
    plt.ylabel('Frequency')
    plt.title('Histogram of omegamh2')
    plt.legend()
    plt.savefig('hist_omegamh2.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

#omegabh2
    plt.hist(data[:, 1], bins=20, label='omegabh2', color='orange')
    plt.xlabel('omegabh2')
    plt.ylabel('Frequency')
    plt.title('Histogram of omegabh2')
    plt.legend()
    plt.savefig('hist_omegabh2.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

#H0
    plt.hist(data[:, 2], bins=20, label='H0', color='green')
    plt.xlabel('H0')
    plt.ylabel('Frequency')
    plt.title('Histogram of H0')
    plt.legend()
    plt.savefig('hist_H0.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

# Scatter plots 
# omegamh2 vs omegabh2

    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('omegamh2')
    plt.ylabel('omegabh2')
    plt.title('Scatter plot of omegamh2 vs omegabh2')
    plt.savefig('scatter_omegamh2_omegabh2.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

# omegamh2 vs H0
    plt.scatter(data[:, 0], data[:, 2])
    plt.xlabel('omegamh2')
    plt.ylabel('H0')
    plt.title('Scatter plot of omegamh2 vs H0')
    plt.savefig('scatter_omegamh2_H0.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

# omegabh2 vs H0
    plt.scatter(data[:, 1], data[:, 2])
    plt.xlabel('omegabh2')
    plt.ylabel('H0')
    plt.title('Scatter plot of omegabh2 vs H0')
    plt.savefig('scatter_omegabh2_H0.png')
    plt.show()
    plt.clf()  # Clear the figure after showing

#write some code to save a nicely formatted text table of the numerical values that you find.
    #with open('data_analysis.txt', 'w') as f:
  
    with open('data_analysis.txt', 'w') as f:               
        f.write(f"|{'Parameter':^12}|{'Mean':^12}|{'Std Dev':^12}|\n")
        f.write(f"|{'-'*12}|{'-'*12}|{'-'*12}|\n")
        f.write(f"|{'omegamh2':^12}|{mean_omegamh2:^12.6f}|{std_omegamh2:^12.6f}|\n")
        f.write(f"|{'omegabh2':^12}|{omegabh2:^12.6f}|{std_omegabh2:^12.6f}|\n")
        f.write(f"|{'H0':^12}|{mean_H0:^12.6f}|{std_H0:^12.6f}|\n")
   
#| (pipe): Separates columns visually, making the table look like a grid.
#^12: Centers the text or number in a field 12 characters wide.
#*12: Repeats the dash character 12 times to make a horizontal line for each column.
#.6f: Formats a floating-point number to 6 decimal places.
#f"...": An f-string, which allows you to insert variables and formatting directly into the string.
#\n: Starts a new line after each row.

task4_(data)

# Write a main function that shows off the use of all your other functions
# this will demonstrate their use and ideally showing that they produce a correct answer. 
# It should run only when the file is run, not when it is imported
 
if __name__ == "__main__":
    print("Task 1a:")
    print("Array of zeros (x):", x)
    print("Array with range of values (y):", y)
    print("2D array of random numbers (z):", z)
    
    print("\nTask 1c:")
    task1c_(y, z)
    
    print("\nTask 1d:")
    task1d_(a, x)
    
    print("\nTask 2a:")
    M = task2a_(3)
    print(M)
    
    print("\nTask 2b:")
    row_sums = task2b_(M)
    print(row_sums)
    
    print("\nTask 3:")
    log_likelihood = task3_(d, mu, sigma)
    print(log_likelihood)
    
    print("\nTask 4:")
    task4_(data)

#"\n" means "new line," so the label appears on a new line, making the output easier to read.
#"Task 1c:" is a label that tells you which part of your code or which function’s output is being shown.






