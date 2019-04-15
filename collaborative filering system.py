import numpy as np
from sklearn.metrics import jaccard_similarity_score

class CFS:

    def __init__(self, jokeID, ratings, k, n):
        self.jokeID = jokeID
        self.ratings = ratings
        self.k = k  # number of knn
        self.n = n  # number of recommend joke
        self.userDict = {}
        self.ItemUser = {}
        self.neighbors = []
        self.recommandList = []
        self.cost = 0.0


    def recommendByUser(self, userId):
        self.formatRate()
        self.n = len(self.userDict[userId])
        self.getNearestNeighbor(userId)
        self.getrecommandList(userId)
        self.getPrecision(userId)

    # Calculate the similarity based on jaccard
    def jaccardCount(self, y_true, y_pred):
        return jaccard_similarity_score(y_true, y_pred)

    # Calculate the similarity based on Cos

    def cosineCount(self, vector1, vector2):
        cosin = np.dot(vector1, vector2) / \
            (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
        return cosin


    
    def getNearestNeighbor(self, userId):
        neighbors = []
        self.neighbors = []
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if(j != userId and j not in neighbors):
                    neighbors.append(j)
        for i in neighbors:
            dist = self.cosineCount(userId, i)
            self.neighbors.append([dist, i])
        # 排序默认是升序，reverse=True表示降序
        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.k]


    def getrecommandList(self,userId):
        self.recommandList = []
        
        recommandDict = {}
        for neighbor in self.neighbors:
            movies = self.userDict[neighbor[1]]
            for movie in movies:
                if(movie[0] in recommandDict):
                    recommandDict[movie[0]] += neighbor[0]
                else:
                    recommandDict[movie[0]] = neighbor[0]

        
        for key in recommandDict:
            self.recommandList.append([recommandDict[key], key])
        self.recommandList.sort(reverse=True)
        self.recommandList = self.recommandList[:self.n]




    
