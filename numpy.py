# -*- coding: utf-8 -*-
"""Copy of NumpyTasks.ipynb



Каждая ячейка ниже представляет из себя одно задание.
В каждом задании необходимо сделать две вещи:


1.   Реализовать функцию, которая делает то, что написано в условии задания
2.   Дополнить авторский тест своими. Не нужно переусердствовать - тесты должны быть емкими и полными, т.е. покрывать corner case(особые случаи) задачи, и все

В этом ноутбуке запрещено пользоваться циклами и list/expression comprehension, map и т.д.. Все операции должны быть векторными.

# Easy - 1 point per task
"""

# Import numpy as np and print the version of the library
import numpy as np
print(np.__version__)

import sys
assert 'numpy' in sys.modules and 'np' in dir()

from numpy.testing import assert_equal


def dummy_example():
    return np.array([1, 2]) 

assert_equal(dummy_example(), [1, 2])

# Create array filled with zeros of size 10
def create_zero_arr():
  return np.zeros(10)

assert_equal(create_zero_arr(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Create matrix with values from 0 to 8 of shape (3, 3)
def create_dummy_matrix():
  return np.arange(9).reshape(3, 3)

assert_equal(create_dummy_matrix(), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# Create matrix of shape (N, N) with 1 on the border and 0 otherwise
def create_ramochka(N):
    a=np.zeros((N-2,N-2))
    return np.pad(a,((1,1),(1,1)), mode='constant', constant_values=1)

assert_equal(create_ramochka(3), [[1, 1, 1], [1, 0, 1], [1, 1, 1]])

# Create matrix of shape (N, N) with сheckboard pattern
def create_checkboard(N):
  a=np.zeros(N*N)
  a[::2]=1
  a=a.reshape(N,N)
  return a


assert_equal(create_checkboard(3), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])

# Create numpy array given start value, step and array length
def create_arr(start, step, length):
    end=start+step*length
    return np.arange(start,end,step)

assert_equal(create_arr(5, 3, 4), [5, 8, 11, 14])

# Extract all even numbers from array
def extract_even(arr):
  arr=np.array(arr)
  return arr[arr % 2 == 0]

assert_equal(extract_even([1, 2, 3, 4]), [2, 4])

# Replace all even numbers with -1
def replace_even(arr):
    arr=np.array(arr)
    arr[arr % 2 == 0]=-1
    return arr

assert_equal(replace_even([1, 2, 3]), [1, -1, 3])

# Convert 1D array of size N to 2D array of shape (sqrt(N), sqrt(N))
def quadratize(arr):
    arr=np.array(arr)
    arr=arr.reshape(int(np.sqrt(arr.size)),int(np.sqrt(arr.size)))
    return arr

assert_equal(quadratize([1, 2, 3, 4]), [[1, 2], [3, 4]])

# Convert 1D array of size N to 2D array of shape (N, N) repeating each element along the row
def row_repeatizer(arr):
  arr=np.array(arr)
  n=arr.size
  arr=np.repeat(arr, arr.size)
  arr=arr.reshape(n,n)
  return arr

assert_equal(row_repeatizer([1, 2, 3]), [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# Perform set operations over 1D arrays and sort the result in ascending way
def intersection(arr1, arr2):
    arr=np.intersect1d(arr1,arr2)
    return arr

def union(arr1, arr2):
    arr=np.union1d(arr1,arr2)
    return arr

def difference(arr1, arr2):
    arr=np.setdiff1d(arr1,arr2)
    return arr

def symmetric_difference(arr1, arr2):
    arr=np.setxor1d(arr1,arr2)
    return arr

assert_equal(intersection([1, 2, 3], [4, 1, 2]), [1, 2])
assert_equal(union([1, 2, 3], [4, 1, 2]), [1, 2, 3, 4])
assert_equal(difference([1, 2, 3], [4, 1, 2]), [3])
assert_equal(symmetric_difference([1, 2, 3], [4, 1, 2]), [3, 4])

# Extract all numbers (and their indices) from array that are not greater than 65 and not less than 18

def pensionazier(arr):
    a = np.vstack((np.array(arr), np.arange(0, np.array(arr).size)))
    return a[:,(a[0]>= 18) & (a[0] <= 65)]

assert_equal(pensionazier([17, 20, 5, 18, 66]), [[20, 18], [1, 3]])



# Swap first and second column in matrix
def column_swapper(arr):
    arr=np.array(arr)
    arr[:,[0, 1]] = arr[:,[1, 0]]
    return arr

assert_equal(column_swapper([[1, 2], [3, 4]]), [[2, 1], [4, 3]])

# Create random matrix NxN containing random values between 1 and 3
def random_matrix(N):
    return np.random.randint(1,4,(N,N))

matrix = random_matrix(3)
assert_equal((matrix >= 1) & (matrix <= 3), np.ones_like(matrix).astype(np.bool))
del matrix

# Compute max for each row of a matrix
def row_max(arr):
    return np.max(arr,axis=1)

assert_equal(row_max([[1, 2], [17, 14]]), [2, 17])

# Compute euclidean distance between two arrays
def eucled_distance(arr1, arr2):
    return np.linalg.norm(np.array(arr1)-np.array(arr2))

assert_equal(eucled_distance([1, 2, 3, 4, 5], [4, 5, 6, 7, 8]), 6.7082039324993694)

"""# Medium - 2 points per task"""

# Write a function that takes scalar X and calculates CDF(X) for gaussian distribution N(0, 1)
# Vectorize this function using numpy
from numpy.testing import assert_almost_equal
from scipy.stats import norm

def cdf(x):
    return norm.cdf(x)

def vectorized_cdf(x):
  y = np.vectorize(cdf)
  return y(x)

assert_almost_equal(vectorized_cdf([-1, 0, 1]), [0.15866, 0.5, 0.84134], decimal=5)

# (just run this) Download IRIS dataset to your local computer and import one column from there with sepal lengths
!wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O iris.data
sepal_length = np.genfromtxt('iris.data', delimiter=',', dtype='object', usecols=[0]).astype(np.float)

# Convert IRIS data to 2D array of numbers omitting the last column (species) with text
    x = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3], dtype=None)
    y = iris_1d.reshape(-1, 4)

# Calculate mean, median, standard deviation of sepal_length

def sepal_statistic(arr):
    return np.array([np.mean(arr, axis=0), np.median(arr, axis=0), np.std(arr, axis=0)])

assert_almost_equal(sepal_statistic(sepal_length), [5.84333333333, 5.8, 0.825301291785])

# Normalize sepal_length such that all values lie between 0 and 1
def sepal_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

normalized = sepal_normalize(sepal_length)
assert_equal((normalized >= 0) & (normalized <= 1), np.ones_like(normalized).astype(np.bool))

# Find 25%, 50%, 75% percentile of sepal length
def sepal_percentile(arr):
    return np.array([np.percentile(arr, 25, axis=0), np.percentile(arr, 50, axis=0), np.percentile(arr, 75, axis=0)])

assert_almost_equal(sepal_percentile(sepal_length), [5.1, 5.8, 6.4])

# Insert 10 np.nan values into sepal_length_with_nan array and print their indices
sepal_length_with_nan = sepal_length.copy()
sepal_length_with_nan = np.concatenate((sepal_length_with_nan, np.array([np.nan]*10)))
print(np.array(np.where(np.isnan(sepal_length_with_nan)))[0])

assert_equal(np.sum(np.isnan(sepal_length_with_nan)), 10)

# Drop np.nan values from sepal_length_with_nan and save the result to sepal_length_without_nan
# DO NOT MODIFY sepal_length_with_nan !!!
sepal_length_without_nan = sepal_length_with_nan[~np.isnan(sepal_length_with_nan)]

assert_equal(sepal_length, sepal_length_without_nan)

# Replace all np.nan with -1 in sepal_length_with_nan and save the result to sepal_length_replaced

sepal_length_replaced = np.nan_to_num(sepal_length_with_nan, nan=-1)

assert_equal(np.isnan(sepal_length_with_nan), sepal_length_replaced == -1)

# Drop all duplicates from array and sort the result in ascending way
def uniquezier(arr):
    return np.unique(arr)

assert_equal(uniquezier([3, 1, 2, 2, 3, 1, 4]), [1, 2, 3, 4])

# Find the most frequent element in array (in case of several possible answers return the lowest number)
def most_frequent(arr):
  arr=np.array(arr)
  return np.unique(arr)[np.argmax(np.unique(arr, return_counts=True)[1])]
                                  

assert_equal(most_frequent([1, 2, -4, 5, -4, 2, -4, 2]), -4)

# Clip all the values in array to interval from 1e-3 to 1
def clipper(arr):
    arr=np.array(arr)
    return np.clip(arr, 1e-3, 1)

assert_equal(clipper([0, 1e-3, 3]), [1e-3, 1e-3, 1])

# Get the positions of top K largest values in array
def k_largest_indices(arr, k):
    arr = np.array(arr)
    return arr.argsort()[-k:][::-1]

assert_equal(k_largest_indices([1, -2, 4, 56, 17, 0], 3), [3, 4, 2])

# Compute min-by-max for rows of matrix.
# Min-by-max is a transform of a matrix where each row is replaced by min in the tow divided by max in the row
def row_min_by_max(arr):
    return np.apply_along_axis(lambda q: np.min(q)/np.max(q), arr=arr, axis=1)

assert_equal(row_min_by_max([[1, 2], [15, 3]]), [0.5, 0.2])

# Subtract 1D array from 2D array such that first element subtracts from the first row, second element for the second row and etc.
def row_subtract(arr, matrix):
    arr = np.array(arr)
    matrix = np.array(matrix)
    arr = np.zeros(matrix.shape) + np.transpose(arr)[:, np.newaxis]
    return matrix - arr

assert_equal(row_subtract([1, 2, 3], [[3, 3, 3], [4, 4, 4], [5, 5, 5]]), 2 * np.ones((3, 3)))

# Compute (A + B) / A where A,B are arrays in-place (with no copy). Compute in-place of A
# NOTE: return (a + b) / a is not a correct solution !
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
b /= a
a /= a
a += b

assert_equal(a, [5, 3.5, 3])
del a
del b

# Consider matrix of shape (N, 2) as a number of cartesian points
# Convert these points to polar points
def to_polar(arr):
    arr=np.array(arr)
    arr = np.array(arr)
    x,y = arr[:,0], arr[:,1]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    return np.hstack((r[:, np.newaxis],t[:, np.newaxis]))

assert_almost_equal(to_polar([[1, 0], [0, 1]]), [[1, 0], [1, np.pi / 2]])

# Find element in array closest to given value. In case of duality return the first one
def find_closest(arr, value):
    arr=np.array(arr)
    return arr[(np.abs(arr-value)).argmin()]

assert_equal(find_closest([1, 2, 3], 1.7), 2)

"""# Hard - 3 points per task"""

# Find duplicate entries (2nd occurrence onwards) in array and mark them as True
# First time occurrences should be False.
from numpy.testing import assert_almost_equal
def mark_duplicates(arr):
    arr=np.array(arr)
    out = np.full(arr.shape[0], True) 
    unique_positions = np.unique(arr, return_index=True)[1] 
    out[unique_positions] = False 
    return out

assert_equal(mark_duplicates([0, 0, 3, 0, 2, 4, 2, 2]), [False, True, False, True, False, False, True, True])

# Challenging
# Find all local maxima in 1D array. Local maximum is an element, surrounded by lesser elements from both sides
def find_local_maxima(arr):
  arr = np.array(arr)
  doublediff = np.diff(np.sign(np.diff(arr)))
  peak_locations = np.where(doublediff == -2)[0] + 1
  return peak_locations

assert_equal(find_local_maxima([1, 3, 7, 1, 2, 6, 0, 1]), [2, 5])

# Make array IMMUTABLE immutable
IMMUTABLE = np.array([1, 2, 3])
IMMUTABLE.flags.writeable = False

try:
    IMMUTABLE[2] = 1
except ValueError:
    print('Well done')
else:
    print('Try again')

# Construct Cauchy matrix from arrays X and Y
# c[i, j]  = 1 / (x[i] - y[j])
def cauchy_matrix(X, Y):
  X=np.array(X)
  Y=np.array(Y)
  lenx = len(X)
  leny = len(Y)
  xm = np.repeat(X,leny)
  ym = np.tile(Y,lenx)
  cauchym = (1.0/(xm-ym)).reshape([lenx,leny]);
  return cauchym

assert_almost_equal(cauchy_matrix([1, 2], [3, 6]), [[-0.5, -0.2], [-1.0, -0.25]])

"""# You are at your own :) - 4 points per task"""

# Compute number of unique colors in an image. Image is an array of shape (W, H, 3) containing integers, color is a tuple of 3 values, e.g. image[i, j] is a color

# Given image compute sum over first two axis (at once). Explain the meaning of the result
def sum_over_two_axis(image):
  image=np.array(image)
  return np.sum(image, axis=(0,1))

# Given 1D array insert zero between each pair of elements
def zero_between(arr):
    arr=np.array(arr)
    return np.dstack((arr,np.zeros_like(arr))).reshape(arr.shape[0],-1)

assert_equal(zero_between([[1, 2, 6],[3, 4, 2],[4, 5, 6]]),[[1,0, 2, 0, 6, 0],[3, 0 ,4 ,0, 2, 0],[4, 0, 5, 0, 6, 0]])

# Just simply compute matrix rank :)
def matrix_rank(arr):
    arr=np.array(arr)
    return  np.linalg.matrix_rank(arr)

# Given an arbitrary number of arrays, build the cartesian product (every combinations of every item)
def cartesian_product(arrays):
   arrays = [np.asarray(a) for a in arrays]
   shape = map(len, arrays)
   ix = np.indices(shape, dtype=int)
   ix = ix.reshape(len(arrays), -1).T
   for n, arr in enumerate(arrays):
     ix[:, n] = arrays[n][ix[:, n]]
   return ix

# Extract from matrix rows with unequal elements
# [[1, 2, 2], [3, 3, 3], [4, 4, 4]] -> [[1, 2, 2]]
def extract_unequal_rows(arr):
    arr = np.array(arr)
    m = arr[1, :].size
    n = arr[:, 1].size
    z = np.repeat(arr[:, 1], m)
    z = z.reshape(n,m)
    x = arr - z
    ind = np.where(np.sum(x, axis=1) != 0)
    arr=np.array(arr[ind])
    return arr


assert_equal(extract_unequal_rows([[1, 5, 2], [3, 3, 3], [4, 4, 4], [2, 3, 6]]),[[1 , 5 , 2],
 [2, 3 , 6]])
