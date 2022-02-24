# Data-410-Project-3

## Multivariate Regression Analysis 
Multivariate models are very similar to univariate models but they intake multiple *x* variables. In general we want: 

![CodeCogsEqn](https://user-images.githubusercontent.com/74326062/155409144-c32228fe-e5ef-4099-ba03-6f41b86da4c4.svg)

which is stating that the expected value of the output variable *y* given certain input *x* variables is about equal to the model **F** with the same *x* input variables.

One of the most important aspects of multivariate models is variable selection. We want the model to properly weight the input *x* variables so that the more important variables have a bigger impact on the output *y* variable. In the case of multivariate regression we have that: 

![CodeCogsEqn-4](https://user-images.githubusercontent.com/74326062/155435647-41993035-a9f6-462c-9d3d-b8ad068aa91c.svg)

We want to solve for the coefficients ![CodeCogsEqn-5](https://user-images.githubusercontent.com/74326062/155435936-041d38a1-ed13-4143-83f4-d4de00fff81c.svg) to know what the best weights for the various *x* inputs are. Solving it out we get that:

![CodeCogsEqn-6](https://user-images.githubusercontent.com/74326062/155436189-218b83ec-3080-406e-b858-12f83661529a.svg)

The code below shows how to implement a multivariate Tricubic Kernel and lowess function. 

```
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X)
    yest = np.zeros(n)
    if len(y.shape)==1:
      y = y.reshape(-1,1)
    if len(X.shape)==1:
      X = X.reshape(-1,1)
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X
    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) 
    # above we compute n vectors of weights
    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],theta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) 
    # above we may have output NaN's where the data points from xnew are OUTSIDE
    # the convex hull of X, if so:
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)]) # OR g(xnew[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
      # above replaces the NaN's with actual values
    return output
```

## Gradient Boosting
One technique to improve upon multivariate lowess regressions is called gradient boosting. In order to improve the model **F** we can train a decision tree where the output is:

![CodeCogsEqn-2](https://user-images.githubusercontent.com/74326062/155410125-b4835286-4c4a-46c9-984c-ed3ad438f4da.svg)

Thus, the gradient boosting model outputs:

![CodeCogsEqn-3](https://user-images.githubusercontent.com/74326062/155410493-827f1d97-7a27-4803-bb59-1cc49089fce4.svg)

 It is likely that this model is more accurate than **F** the simple lowess output.
 
The code below shows how to implement a multivariate gradient boosted lowess function. It uses the regular lowess function showed above.

```
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # first we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) 
  new_y = y - Fx
  tree_model = dtr(max_depth = 3, random_state = 123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output
```
## Extreme Gradient Boosting (XGBoost)
Extreme gradient boosting (XGBoost) is a technique to improve on gradient boosting. XGBoost works by trying to make the decision trees as described above more accurate. We compare splits in the decision tree and make sure each split improves the accuracy of the model. If a split does not improve accuracy that split will not be made. 

XGBoost has several important hyperparameters in order to make the resulting model as accurate as possible. Lambda decrease the sensitivity to individual data points, and therefore outliers and Gamma is the minimum loss required to split a node of the decision trees in the model. 

The final prediction made by XGBoost is determined by the learning rate, and the number of estimators/trees. 

XGBoost has a function in the sklearn library in Python, and therefore no code needs to be written, other than importing the correct functions. Below is an example of calling the XGBoost function. 

```
import xgboost as xgb
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, reg_lambda=20, alpha=1, gamma=10, max_depth=3)
```

## Comparing Models

