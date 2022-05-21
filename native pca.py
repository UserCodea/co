import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class PCA:
    def _init_(self,n_components) :
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self,x):
        self.mean = np.mean(x, axis =0)
        x = x-self.mean
        cov = np.cov(x.T)
        eigenvalues,eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]

    def transform(self,x):
        x = x-self.mean
        return np.dot(x,self.components.T)


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


pca = PCA()
pca.n_components=2
pca.fit(X_train)
X_trasin = pca.transform(X_train)
print(X_trasin)
