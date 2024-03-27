import numpy as np
import pandas as pd

import rerun as rr
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt


def PCA(X, n_components=2) -> np.ndarray:

    # first normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # next compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # next compute the eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)



    # sort the eigenvectors by decreasing eigenvalues
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    # select the top n_components eigenvectors that maximize the variance
    eig_vectors = eig_vectors[:, :n_components]

    # project the data onto the eigenvectors
    X_pca = X.dot(eig_vectors) # linear transformation , shape (n_samples, n_components)

    return X_pca




def update_plot(X,y,regularization=None, alpha=0.0)->None:
    # Perform linear regression with the current parameters
    if regularization == 'ridge':
        model = Ridge(alpha=alpha)
    elif regularization == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(X, y)

    rr.log('y', rr.Points3D(y, colors='red'))

    
    # Plot the data points and the regression plane
    
    x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                                  np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
    
    rr.log('Regression Plane', rr.Points3D(np.c_[x_surf.ravel(), y_surf.ravel(), z_surf.ravel()], colors='green'))

    
    return None





if __name__ == '__main__':
    # Load the data
    rr.init("Linear Regression", spawn=True)
    data = pd.read_csv('data.csv')
    X = data.drop('y', axis=1).values
    y = data['y'].values

    X_pca = PCA(X, n_components=2)

    update_plot(X_pca,y, regularization=None, alpha=0.0)