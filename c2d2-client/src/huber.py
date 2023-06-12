import numpy as np
from sklearn.linear_model import HuberRegressor

heatmap = np.load('heatmap.npy')
# Input data
X = np.arange(len(heatmap)).reshape(-1, 1)

# Output data
y = heatmap.reshape(-1, 1)
huber = HuberRegressor().fit(X, y)
y_pred = huber.predict(X)
line = y_pred.reshape(heatmap.shape)
line[line < 0] = 0
line[line > np.max(line)/2] = 0



import numpy as np
from sklearn.linear_model import HuberRegressor

heatmap = heatmap
x, y = np.nonzero(heatmap)

epsilon = 1.4
regressor = HuberRegressor(alpha=0.0, epsilon=epsilon)
regressor.fit(x.reshape(-1, 1), y)
# Find the edges of the heatmap
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# Find the x-coordinates of the edges
xstart = xmin if np.abs(regressor.predict([[xmin]]) - ymin) < np.abs(regressor.predict([[xmin]]) - ymax) else xmax
xend = xmax if np.abs(regressor.predict([[xmax]]) - ymin) < np.abs(regressor.predict([[xmax]]) - ymax) else xmin

# Use the regression model to predict the y-coordinates
ystart = int(round(regressor.predict([[xstart]])[0]))
yend = int(round(regressor.predict([[xend]])[0]))
