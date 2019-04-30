import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


df = pd.read_excel('jester-data-3.xls',header=None)
# Data from 24,938 users who have rated between 15 and 35 jokes, a matrix with dimensions 24,938 X 101.

print(df)
