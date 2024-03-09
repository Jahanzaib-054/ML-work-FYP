import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

# x = np.array([[1,2,3],[4,5,6]])
# print("x:\n{}".format(x))
# eye = np.eye(4)
# sparse_matrix = sparse.csr_matrix(eye)

# data = np.ones(4)
# row_indices = np.arange(4)
# col_indices = np.arange(4)
# eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
# print("Numpy: array:\n{}".format(eye))
# print("\nScipy sparse CSR matrix:\n{}".format(sparse_matrix))
# print("COO representation:\n{}".format(eye_coo))

# x = np.linspace(-10, 10, 100)
# y = np.sin(x)
# plt.plot(x, y, marker="x")
# plt.show()

data = {'Name':["john", "anna", "jerry", "hilda"],
        'location':["New York", "Paris", "Sialkot", "Miami"],
        'Age':[18, 21, 23, 15]
        }
data_pandas = pd.DataFrame(data)
print("pandas dataframe:\n{}".format(data_pandas[data_pandas.Age > 18]))