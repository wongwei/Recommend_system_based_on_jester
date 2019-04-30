import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from surprise import Reader
from surprise import SVD

ratings = pd.read_csv('/Users/wangwei/Recommender System/jester_dataset_2/jester_ratings.dat',
                      sep=2*'\t', names=['userid', 'jokeid', 'ratings'])

reader = Reader(line_format='user item rating', rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings[['userid', 'jokeid', 'ratings']],reader)
# # We can now use this dataset as we please, e.g. calling cross_validate
# cross_validate(NormalPredictor(), data, cv=2)
# We'll use the famous SVD algorithm.
# could be replaced by NormalPredictor() BaselineOnly()  KNNBasic() KNNWithMeans() and so on 
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# It takes really long time to train the model once we used the SVD algorithem
# As I try to use implement knn it took huge memory not only in the memory card but also disk

#gridsearchcv on training set to find best params

param_grid = {'reg_all': [0.02, 0.1, 1, 10, 100, 10000]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(ratings)
# combination of parameters that gave the best RMSE score
# best RMSE score
print('the best socre is {0}, the best params is {1}'.format(gs.best_score['mae'],gs.best_params['mae']))


