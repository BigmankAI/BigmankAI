import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#read data from spreadsheet
df = pd.read_excel('Whole_State_Data2.xlsx')
X = df[['Year']]
Y = df[['NOx']]

#train using the first 8 datapoints
train_x = X[:8]
train_y = Y[:8]

#test against the last 2
test_x = X[8:]
test_y = Y[8:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training data
regr.fit(train_x, train_y)

# Generate a forecast using the testing data
pred_y = regr.predict(test_x)

# Coefficients and mean squared error for accuracty determination
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(test_y, pred_y))
print("Coefficient of determination: %.2f" % r2_score(test_y, pred_y))


#plot the forecast and compare with actual
plt.scatter(X, Y)
plt.plot(test_x, pred_y)
plt.show()

