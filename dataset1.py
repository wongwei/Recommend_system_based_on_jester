import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


df = pd.read_excel('jester-data-1.xlsx',header=0)

pca = PCA(n_components=2)
pca.fit(df)
print(df)
