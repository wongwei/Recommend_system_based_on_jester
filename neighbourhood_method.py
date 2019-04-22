import numpy as np
from sklearn.metrics import jaccard_similarity_score
import operator


# neighbourhood method

class CFS1:

    def __init__(self, ratings, userId, n,threshold):
        # self.jokeID = jokeID
        self.ratings = ratings
        # self.k = k  # number of knn
        self.n = n  # number of recommend joke
        self.cosin = {}
        self.userDict = {}
        self.ItemUser = {}
        self.neighbors = []
        self.recommandList = []
        self.threshold = threshold
        self.cost = {}
        self.userId = userId

    def createUserDict(self):
        i = 1
        while i < len(self.ratings):
            self.userDict[i] = self.ratings[i]
            i += 1
        # print("this is a test",self.ratings[41534])

    def recommendByUser(self):
        # self.n = len(self.userDict[userId])
        self.getNearestNeighbor(self.userId)
        self.getrecommandList(self.userId)
        return self.recommandList
    # # Calculate the similarity based on jaccard
    # def jaccardCount(self, user1, user2):
    #     return jaccard_similarity_score(user1, user2)

    # Calculate the similarity based on Cos

    def cosineCount(self):
        self.createUserDict()
        vector1 = self.userDict[self.userId]
        for key in self.userDict:
            self.cosin[key] = np.dot(vector1, self.userDict[key]) / \
                (np.linalg.norm(vector1)*(np.linalg.norm(self.userDict[key])))

    def getNearestNeighbor(self, userId):
        self.cosineCount()
        # print(self.cosin[1])
        sorted_cosin = sorted(self.cosin.items(),
                              key=lambda kv: kv[1], reverse=True)
        # return value start from the second one cause the most similarity one is itself
        return sorted_cosin[1:self.n]

    # get the sorted values mean that we can get the most similarity users of our specific one
    # next we can find out those item without ratings (0 marks) and recommend the items base on similarity users choice

    def getrecommandList(self, userId):
        neighborId = []
        sorted_cosin = self.getNearestNeighbor(self.userId)
        # print(sorted_cosin)
        for i in range(len(sorted_cosin)):
            neighborId.append(sorted_cosin[i][0])
    # we have to check the empty value(0) of our test user
    # change the form of ratings
        # convert the ndarray to list
        ratingOfTester = np.array(self.ratings[self.userId]).tolist()
        userIndices = []
        for indice, value in enumerate(ratingOfTester):
            if value == 0:
                userIndices.append(indice)
        # check the ratings of those similarity user of those unrating jokes

        neighbourCorrespondingDict = {}

        for i in userIndices:
            neighbourRating = []
            for j in neighborId:
                neighbour = np.array(self.ratings[j]).tolist()
                neighbourRating.append(neighbour[i])
            neighbourCorrespondingDict[i] = neighbourRating

        for key,value in neighbourCorrespondingDict.items():
            count = 0
            for item in value:
                if item > self.threshold:
                    count +=1
            if count > 1:
                self.recommandList.append(key)

