# Visualization Linear Regression w/Rerun + PCA


I've been revieweing ML and linear regression for a bit so I wanted to create another tool that could grant some visual intuition for regression tasks. Obviously a 2D graph is practical but I wanted to try out a PointCloud approach and visualize OLS in 3 dimensions.

To accomplish this, I first PCA the covariates down to two components and then do a simple OLS to find the "plane" that minimizes the residuals squared error.

I'm in the process of adding regularization and allowing realtime adjustments of alpha.
