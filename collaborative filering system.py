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

    def createUserDict(self):
        i = 1
        while i < len(self.ratings):
            self.userDict[i] = self.ratings[i]
            i += 1
        

    def recommendByUser(self, userId):
        self.n = len(self.userDict[userId])
        self.getNearestNeighbor(userId)
        self.getrecommandList(userId)

    # # Calculate the similarity based on jaccard
    # def jaccardCount(self, user1, user2):
    #     return jaccard_similarity_score(user1, user2)

    # Calculate the similarity based on Cos

    def cosineCount(self,userID):
        vector1 = self.userDict[userID]
        for key in self.userDict:
            cosin = np.dot(vector1,self.userDict[key]) / \
                (np.linalg.norm(vector1)*(np.linalg.norm(self.userDict[key])))
        return cosin

    def getNearestNeighbor(self, userId):
        neighbors = []
        self.neighbors = []
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if(j != userId and j not in neighbors):
                    neighbors.append(j)
        for i in neighbors:
            dist = self.cosineCount(userId)
            self.neighbors.append([dist, i])
        
        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.k]


    def getrecommandList(self,userId):
        self.recommandList = []
        
        recommandDict = {}
        for neighbor in self.neighbors:
            jokes = self.userDict[neighbor[1]]
            for joke in jokes:
                if(joke[0] in recommandDict):
                    recommandDict[joke[0]] += neighbor[0]
                else:
                    recommandDict[joke[0]] = neighbor[0]

        
        for key in recommandDict:
            self.recommandList.append([recommandDict[key], key])
        self.recommandList.sort(reverse=True)
        self.recommandList = self.recommandList[:self.n]

    


    
