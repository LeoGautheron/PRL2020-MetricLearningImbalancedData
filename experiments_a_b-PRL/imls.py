
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np
from lmnn import LargeMarginNearestNeighbor as LMNN


class imls():
    def __init__(self, k=3, mu=0.5, coef=5, randomState=np.random):
        self.coef = coef
        self.k = k
        self.mu = mu
        self.randomState = randomState

    def fitPredict(self, Xtrain, ytrain, Xtest):
        while self.coef*self.k > Xtrain.shape[0]:
            self.coef -= 1
        nn = NearestNeighbors(n_neighbors=self.coef*self.k)
        nn.fit(Xtrain)
        nearestNeighbors = nn.kneighbors(Xtest, return_distance=False)
        lmnn = LMNN(k=self.k,randomState=self.randomState, mu=self.mu)
        lmnn.fit(Xtrain, ytrain)
        Xtrain = lmnn.transform(Xtrain)
        Xtest = lmnn.transform(Xtest)
        nn = NearestNeighbors(n_neighbors=self.coef*self.k)
        nn.fit(Xtrain)
        newNearestNeighbors = nn.kneighbors(Xtest, return_distance=False)
        matching = np.array([len(np.intersect1d(
               nearestNeighbors[i],newNearestNeighbors[i]))>=int(self.coef*0.8)
                             for i in range(len(nearestNeighbors))])
        while matching.all() == False:
            nearestNeighbors = newNearestNeighbors.copy()
            lmnn = LMNN(k=self.k,randomState=self.randomState, mu=self.mu)
            lmnn.fit(Xtrain, ytrain)
            Xtrain = lmnn.transform(Xtrain)
            Xtest = lmnn.transform(Xtest)
            nn = NearestNeighbors(n_neighbors=self.coef*self.k)
            nn.fit(Xtrain)
            newNearestNeighbors = nn.kneighbors(Xtest, return_distance=False)
            matching = np.array([len(np.intersect1d(
              nearestNeighbors[i], newNearestNeighbors[i]))>=int(self.coef*0.8)
                                 for i in range(len(nearestNeighbors))])
        knc = KNeighborsClassifier(n_neighbors=self.k)
        knc.fit(Xtrain, ytrain)
        return knc.predict(Xtest)
