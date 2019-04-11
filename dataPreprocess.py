import pandas as pd
import numpy as np 

# dataframe1 = pd.read_excel('jester-data-1.xls', sep="\t", name = list(range(1,100)),header = list(range(1,100)))
#dataframe1.insert(0,"USERS",)
# dataframe2 = pd.read_excel('jester-data-2.xls', index_col=0)
# dataframe3 = pd.read_excel('jester-data-3.xls', index_col=0)
# nullCount = dataframe1.isnull().sum()/len(dataframe1)*100
# print(dataframe1.head(0))
# print(dataframe1.iloc[:10, :10])
# print(nullCount)

# Description of the rating file: without the rating of some jokes 


jokes = pd.read_csv('/Users/wangwei/Recommender System/jester_dataset_2/jester_items.dat',sep='\t',error_bad_lines=False)
ratings = pd.read_csv('/Users/wangwei/Recommender System/jester_dataset_2/jester_ratings.dat',sep=2*'\t',names=['userid','jokeid','ratings'],lineterminator='\n')

# in the rating file, the space between to column is 2 tab, so i double the sep
ratings=ratings.pivot_table(index='userid', columns='jokeid', values='ratings') #reshape the dataframe to a easy under standing matrix

print(ratings.loc[1,5])

