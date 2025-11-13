import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
  input_length = len(input_array)
  kernel_length = len(kernel_array)
  return (input_length - kernel_length + 1)


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
  size = compute_output_size_1d(input_array, kernel_array)
  output_array = [0]*size
  for i in range(size):
    output_array[i] = input_array[0 + i]*kernel_array[0] + input_array[1 + i]*kernel_array[1] + input_array[2 + i]*kernel_array[2]
  return output_array

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
  input_height = len(input_matrix[:, 0])
  kernel_height = len(kernel_matrix[:, 0])
  input_width = len(input_matrix[0])
  kernel_width = len(kernel_matrix[0])
  return (input_height - kernel_height + 1, input_width - kernel_width + 1)

# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
  output_height, output_width = compute_output_size_2d(input_matrix, kernel_matrix)
  output_matrix = np.zeros((output_width, output_height))
  for i in range(output_width):
    for j in range(output_height):
      patch = input_matrix[i:i + kernel_matrix.shape[0], j:j + kernel_matrix.shape[1]]
      output_matrix[i, j] = np.sum(patch * kernel_matrix)
  return output_matrix


# -----------------------------------------------
