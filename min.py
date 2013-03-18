from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
import matplotlib.pyplot as plt

def reshape_array(array):
        return np.reshape(array, (array.shape[0], ))

x = np.genfromtxt("data.csv", delimiter=",")[:,1]
y = np.genfromtxt("data.csv", delimiter=",")[:, 2]
err =  np.genfromtxt("data.csv", delimiter=",")[:, 3]

y = reshape_array(y)
x = reshape_array(x)
err = reshape_array(err)
c = np.zeros((y.shape[0], y.shape[0]))




print c.shape
for i in range(0, c.shape[0]):
  print err[i]
  c[i, i] = err[i]
  
  
params = (0.4, 4)  


  
C = np.matrix(c)
print C.diagonal()
Y = np.matrix(y)
A = np.ones((x.shape[0], 2))
A[:, 1] = x
print A.shape

def getX(params):
  #params = np.array()
  X = np.matrix([[params[0]], [params[1]]])
  return X
#params = [0.5, 3, 29]

#fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2


#objective_function = lambda params: np.sum(np.divide((y - model_2(x, y, params))**2, err**2))

objective_function_linalg = lambda params: np.dot(np.dot(Y - np.dot(A,getX(params)).getT(), C.getI()), np.matrix(Y-A*getX(params)))

def model_1(x, y, params):
  return params[0]*x+params[1]

def model_2(x, y, params):
    return params[0]*x**2 + params[1]*x + params[2]



#res = minimize(objective_function(x, y, err))


print ((Y - np.dot(A,getX(params))).T).shape
print Y.shape, A.shape, getX(params).shape, 'X', np.dot(A,getX(params)).shape, 




res = minimize(objective_function_linalg, params)
print res.x
#print np.dot(A, getX(params))

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, c='k')
ax.errorbar(x, y, yerr=err, linestyle = "none", color="black") 
ax.scatter(x, model_2(x, y, res.x), c='red')
#ax.scatter(x, model_1(x, y, res.x), c='k')
#ax.plot
ax.text(20, 500,"y ="+str(round(res.x[0], 2))+"x+"+str(round(res.x[1], 2)))  
print x.shape
#plt.show()


'''