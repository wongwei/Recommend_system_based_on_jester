import numpy as np
from sklearn.metrics import jaccard_similarity_score
import operator

class CFS:

    def __init__(self,ratings,userId,k):
        # self.jokeID = jokeID
        self.ratings = ratings
        self.k = k  # number of knn
        # self.n = n  # number of recommend joke
        self.cosin = {}
        self.userDict = {}
        self.ItemUser = {}
        self.neighbors = []
        self.recommandList = []
        self.cost = {}
        self.userId = userId

    def createUserDict(self):
        i = 1
        while i < len(self.ratings):
            self.userDict[i] = self.ratings[i]
            i += 1
        

    def recommendByUser(self):
        # self.n = len(self.userDict[userId])
        self.getNearestNeighbor(self.userId)
        # self.getrecommandList(userId)

    # # Calculate the similarity based on jaccard
    # def jaccardCount(self, user1, user2):
    #     return jaccard_similarity_score(user1, user2)

    # Calculate the similarity based on Cos

    def cosineCount(self):
        self.createUserDict()
        vector1 = self.userDict[self.userId]
        for key in self.userDict:
            self.cosin[key] = np.dot(vector1,self.userDict[key]) / \
                (np.linalg.norm(vector1)*(np.linalg.norm(self.userDict[key])))
        

    def getNearestNeighbor(self, userId):
        self.cosineCount()
        # print(self.cosin[1])
        sorted_cosin = sorted(self.cosin.items(), key=lambda kv: kv[1],reverse=True)
        return sorted_cosin
    # def getrecommandList(self,userId):
    #     self.recommandList = []
        
    #     recommandDict = {}
    #     for neighbor in self.neighbors:
    #         jokes = self.userDict[neighbor[1]]
    #         for joke in jokes:
    #             if(joke[0] in recommandDict):
    #                 recommandDict[joke[0]] += neighbor[0]
    #             else:
    #                 recommandDict[joke[0]] = neighbor[0]

        
    #     for key in recommandDict:
    #         self.recommandList.append([recommandDict[key], key])
    #     self.recommandList.sort(reverse=True)
    #     self.recommandList = self.recommandList[:self.n]

    


    
