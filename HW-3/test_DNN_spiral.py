# test DNN using spiral data. Zhengdao Wang. ISU EE 526X

from DNN import NeuralNetwork,ObjectiveFunction,ReLU,Linear
import numpy as np
import matplotlib.pyplot as plt

def generateData(N=100, K=3, D=2):
  ''' N points in each class. K classes. D dimensions '''
  X = np.zeros((D, N*K)) # data matrix (each row = single example)
  y = np.zeros((K, N*K), dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    y[j,ix] = 1
  return X,y

def plot(X, Y, nn):
  plt.ion()
  fig=plt.figure(1)
  plt.clf()
  x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
  y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)

  plt.scatter(X[0, :], X[1, :], c=np.argmax(Y,axis=0),
    s=40)

  if nn is not None:
    N=150
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
      np.linspace(y_min, y_max, N))
    Z = nn.predict( np.c_[xx.ravel(), yy.ravel()].T )
    Z = np.argmax(Z, axis=0)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,  alpha=0.3)
    fig.canvas.start_event_loop(0.01)

def main():
  D=2 # Input dimension
  Odim=3 # number of outputs
  layers=[ (100, ReLU), (40, ReLU), (Odim, Linear) ]
  nn=NeuralNetwork(D, layers)
  eta=1e-1
  # nn.setOptimizer(method={"name":'SGD', "learning_rate":eta})
  nn.setOptimizer("ADAM")
  nn.setRandomWeights(0.1)
  CE=ObjectiveFunction('crossEntropyLogit')

  N=200 # points per cluster
  K=3 # number of clusters
  X,Y=generateData(N,K,D)

  for i in range(10000):
    logp=nn.doForward(X)
    J=CE.doForward(logp, Y)
    dz=CE.doBackward(Y)
    dx=nn.doBackward(dz)
    nn.updateWeights()
    if (i%100==0):
      print( '\riter %d, J=%f' % (i, J), end='')
      plot(X,Y,nn)
    # nn.print(['W', 'b'])
  input('Press Enter to Finish')

main()

# run: python -Werror %
