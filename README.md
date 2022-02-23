# Data-410-Project-3

## Multivariate Regression Analysis 
Multivariate models are very similar to univariate models but they intake multiple *x* variables. In general we want: 

![CodeCogsEqn](https://user-images.githubusercontent.com/74326062/155409144-c32228fe-e5ef-4099-ba03-6f41b86da4c4.svg)

which is stating that the expected value of the output variable *y* given certain input *x* variables is about equal to the model **F** with the same *x* input variables.

One of the most important aspects of multivariate models is variable selection. We want the model to properly weight the input *x* variables so that the more important variables have a bigger impact on the output *y* variable. 

**more about varibale selection here**

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

Thus, the gradient boosting model:

![CodeCogsEqn-3](https://user-images.githubusercontent.com/74326062/155410493-827f1d97-7a27-4803-bb59-1cc49089fce4.svg)

 It is likely that this model is more accurate than **F** the simple lowess output.

## Extreme Gradient Boosting (XGboost)
