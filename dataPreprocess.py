
import pandas as pd
import numpy as np
import os
from neighbourhood_method import CFS1
from bs4 import BeautifulSoup


with open('/Users/wangwei/Recommender System/jester_dataset_2/jester_items.dat', 'r') as f:
    allJoke = f.read()
soup = BeautifulSoup(allJoke, 'lxml')
jokelist = []
for s in soup.strings:
    jokelist.append(s)  # remove all html tag
# remove the Escape character \n
jokelist = list(map(lambda s: s.strip(), jokelist))
jokeString = list(filter(None, jokelist))  # remove the empty element


# Description of the rating file: without the rating of some jokes
ratings = pd.read_csv('/Users/wangwei/Recommender System/jester_dataset_2/jester_ratings.dat',
                      sep=2*'\t', names=['userid', 'jokeid', 'ratings'])

# in the rating file, the space between to column is 2 tab, so i double the sep
# reshape the dataframe to a easy under standing matrix
ratings = ratings.pivot_table(
    index='userid', columns='jokeid', values='ratings')
ratings_col = ratings.columns.tolist()
newRatings = np.nan_to_num(ratings)
tester = 35
result = CFS1(newRatings, tester, 5, 5)
recomendList = result.recommendByUser()

def searchJokeByIndice(indexvalue):
    recommedjoke = []
    for i in indexvalue:
        recommedjoke.append(ratings_col[i])
    
    # search the string to find out the content of jokes
    startIndex = 0
    endIndex = 0
    for item in recommedjoke:
        flag = item
        for index, value in enumerate(jokeString):
            if value == str(flag)+':':
                startIndex = index
            elif value == str(flag+1)+':':
                endIndex = index
            jokeContent = ' '.join(jokeString[startIndex:endIndex])
        print('the recommeded joke for user {0} is \n {1}'.format(tester,jokeContent))


searchJokeByIndice(recomendList)
