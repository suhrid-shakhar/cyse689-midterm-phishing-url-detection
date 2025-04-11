# Multi-Layer Neural Network Implementation. Zhengdao Wang. 2015

import numpy as np

import pdb
debug=pdb.set_trace

class Linear:
  def forward(self, x): return x
  def backward(self, dy): return dy
  def name(self): return "Linear"

class ReLU:
  ''' ReLU nonlinearity '''
  def forward(self, x):
    y=np.maximum(0, x)
    self.x=x
    return y

  def backward(self, dy):
    dx = (self.x>=0)*dy
    return dx

  def name(self):
    return 'ReLU'

class Layer:
  ''' One neural layer '''

  def __init__(self, nInputs, nOutputs, NL):
    ''' nInputs : number of inputs
        nOutputs: number of outputs
        NL : ReLU, Logistic, or softmax objects
    '''
    self.nInputs=nInputs
    self.nOutputs=nOutputs
    self.W=np.zeros((nOutputs, nInputs))
    self.b=np.zeros((nOutputs, 1))
    self.dW=np.zeros((nOutputs, nInputs))
    self.db=np.zeros((nOutputs, 1))
    self.dZ=None
    self.NL=NL()
    self.setOptimizer(method='ADAM') # default optimizer

  def setRandomWeights(self, M=0.1):
    ''' set random uniform weights of max size M '''
    self.W=np.random.rand(self.nOutputs, self.nInputs)*M
    self.b=np.random.rand(self.nOutputs, 1)*M

  def copyWeightsFrom(self, src):
    ''' copy weight from another layer object of the same config '''
    np.copyto(self.W, src.W)
    np.copyto(self.b, src.b)

  def doForward(self, _input):
    Z=self.W.dot(_input)+self.b
    _output=self.NL.forward(Z)
    self.Z=Z
    self.input=_input
    return _output

  def doBackward(self, dOutput):
    self.dZ=self.NL.backward(dOutput)
    self.dW=self.dZ@self.input.T
    self.db=np.sum(self.dZ, axis=1, keepdims=True)
    dInput=self.W.T@self.dZ
    return dInput

  def setOptimizer(self, method):
    if method=='ADAM':
      self.__setOptimizer_ADAM()
    elif method=='SGD':
      self.__setOptimizer_SGD()
    elif type(method) is dict:
      if method["name"]=="ADAM":
        self.__setOptimizer_ADAM(method["learning_rate"],
          method["beta1"], method["beta2"], method["epsilon"])
      if method["name"]=="SGD":
        self.__setOptimizer_SGD(method["learning_rate"])
    else:
      raise ValueError('method should be either string or dict')

  def __setOptimizer_SGD(self, learning_rate=1E-2):
      self.eta=learning_rate
      self.updateWeights=self.updateWeights_SGD

  def __setOptimizer_ADAM(self, learning_rate=1E-3, beta1=0.9, beta2=0.999,
      epsilon=1E-7):
    self.eta=learning_rate
    self.beta1=beta1
    self.beta2=beta2
    self.epsilon=epsilon
    self.mW=0*self.W
    self.vW=0*self.W
    self.mb=0*self.b
    self.vb=0*self.b
    self.t = 0 # number of iterations
    self.updateWeights=self.updateWeights_ADAM

  def updateWeights_SGD(self):
    self.W -= self.eta*self.dW
    self.b -= self.eta*self.db

  def updateWeights_ADAM(self):
    self.t += 1
    b1=self.beta1
    b2=self.beta2
    eta=self.eta*np.sqrt(1.- np.power(b2, self.t))/(1.- np.power(b1, self.t))
    self.mW = b1*self.mW + (1-b1)*self.dW
    self.vW = b2*self.vW + (1-b2)* np.square(self.dW)
    self.W -= eta*self.mW/(np.sqrt(self.vW)+self.epsilon)
    self.mb = b1*self.mb + (1-b1)*self.db
    self.vb = b2*self.vb + (1-b2)*np.square(self.db)
    self.b -= eta*self.mb/(np.sqrt(self.vb)+self.epsilon)

class NeuralNetwork:
  ''' a neural network '''

  def __init__(self, nInputs, layers, M=1e-1, name=''):
    ''' layers=( (n_neurons, NL), ... )
    '''
    if not isinstance(layers, list) or \
       not isinstance(layers[0], tuple) or \
       not isinstance(layers[0][0], int) or \
       not isinstance(layers[0][0], int):
      raise ValueError('layers must be a list of (nNeuron, NL) tuples')

    self.nLayers=len(layers)
    self.A=[None]*(self.nLayers+1)  # input + all activations
    self.dA=[None]*(self.nLayers+1) # input + all activations
    self.layers=[ Layer(nInputs, layers[0][0], layers[0][1]) ]
    for l in range(1,self.nLayers):
      self.layers.append( Layer(layers[l-1][0], layers[l][0], layers[l][1]) )
    self.setRandomWeights(M)
    self.name=name

  def setRandomWeights(self, M=1e-1):
    for l in range(0,self.nLayers):
      self.layers[l].setRandomWeights(M)

  def copyWeightsFrom(self, src):
    for l in range(0,self.nLayers):
      self.layers[l].copyWeightsFrom(src.layers[l])

  def setOptimizer(self, method):
    for l in range(self.nLayers): self.layers[l].setOptimizer(method)

  def doForward(self, _input):
    self.A[0]=_input  # A = activations
    for l in range(self.nLayers):
      self.A[l+1]=self.layers[l].doForward(self.A[l])
    return self.A[self.nLayers]

  def predict(self, _input):
    return self.doForward(_input)

  def doBackward(self, dOutput):
    self.dA[self.nLayers]=dOutput
    for l in range(self.nLayers,0,-1):
      self.dA[l-1]=self.layers[l-1].doBackward(self.dA[l])
    return self.dA[0]

  def updateWeights(self):
    for l in range(self.nLayers):
      self.layers[l].updateWeights()

  def summary(self):
    print('\n====== SUMMARY: {} ======='.format(self.name))
    for l in range(self.nLayers):
      layer=self.layers[l]
      print('Layer {}: Input: ({},None). Output: ({},None). NL={}'.format(
        l, layer.nInputs, layer.nOutputs, layer.NL.name()) )
    print('====== END   SUMMARY =======')

  def print(self, want=['W', 'dW', 'b', 'db', 'Z', 'dZ', 'Input',
        'Output', 'dInput', 'dOutput']):
    for l in range(self.nLayers):
      Map={'W':self.layers[l].W,
        'dW': self.layers[l].dW,
        'b': self.layers[l].b,
        'db': self.layers[l].db,
        'Z': self.layers[l].Z,
        'dZ': self.layers[l].dZ,
        'Input': self.A[l],
        'Output': self.A[l+1],
        'dInput': self.dA[l],
        'dOutput': self.dA[l+1]}
      print('======== Layer %d =========' % l)
      for k in want:
        print('                           '+k + ':\n', Map[k])

  def save(self, filename):
    import pickle
    with open(filename, 'wb') as f:
      pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

  def load(self, filename):
    import os.path
    import pickle
    if os.path.isfile(filename):
      with open(filename, 'rb') as f:
        nn=pickle.load(f)
      self.__dict__.update(nn.__dict__)

class ObjectiveFunction:
  def crossEntropyLogitForward(self, logits, y):
    ''' Cross entropy between [log(p_1), ..., log(p_n)]+C, and
        [y_1, ..., y_n]. The former vector is called logits in Tensorflow.
        It can be obtained by just W*x+b, without any nonlinearity.
        Input logits and y are both n by m, where n is the number of classes
        and m is the number of data points.
    '''
    a=logits.max(axis=0) # log-sum-exp trick
    logp=(logits-a)-np.log( np.sum(np.exp(logits-a), axis=0) )
    J=-np.sum( logp*y )/y.shape[1]
    self.logp=logp
    return J

  def crossEntropyLogitBackward(self, y):
    dz=np.exp(self.logp)-y
    m=y.shape[1]
    dz*=1/m
    return dz

  def logisticForward(self, logits, y):
    ''' logit is w*x+b, one scalar for each data point.
        Input logits is 1 by m, where m is the number of points.
        Input y is 2 by m, first row y_0, and second row y_1.
    '''
    logits=np.vstack( (np.zeros((1, logits.size)), logits) ) # padding 0 on top
    return self.crossEntropyLogitForward(logits, y)

  def logisticBackward(self, y):
    dz=self.crossEntropyLogitBackward( np.vstack( (1-y,y) ) )
    return dz[1,None]

  def MSE_Forward(self, yhat, y):
    ''' Least squares cost function -- forward. Input yhat and y
        are both 1 by m, where m is the number of data points.
    '''
    self.diff = yhat-y
    return np.mean( np.square(self.diff) )
    # Note that the mean is also done along the output dimension, so it is
    # MSE per output dimension per sample -- this behavior is consistent
    # with Keras

  def MSE_Backward(self, y):
    ''' Least squares cost function -- backward. Uses the stored
        self.diff from the leastSquaresForward function.
    '''
    return 2*self.diff/(y.shape[1]*y.shape[0])

  def __init__(self, name):
    if name=='crossEntropyLogit':
      self.doForward=self.crossEntropyLogitForward
      self.doBackward=self.crossEntropyLogitBackward
    elif name=='logistic':
      self.doForward=self.logisticForward
      self.doBackward=self.logisticBackward
    elif name=='MSE':
      self.doForward=self.MSE_Forward
      self.doBackward=self.MSE_Backward

class Model:
  def __init__(self, nn, loss, optimizer="ADAM", metric=None):
    self.nn=nn
    self.nn.setOptimizer(optimizer)
    self.objective=ObjectiveFunction(loss)
    if metric is None:
      if loss=='crossEntropyLogit':
        metric='accuracy'
      elif loss=='MSE':
        metric='MSE'
    if metric=='accuracy':
      self.doMetric=self.__accuracy
    elif metric=='MSE':
      self.doMetric=self.__MSE

  def copyFromKeras(self, keras_model):
    nn=self.nn
    kw=keras_model.get_weights()
    for l in range(0,nn.nLayers):
      bs=nn.layers[l].b.shape
      nn.layers[l].W=np.copy(kw[2*l].T)
      nn.layers[l].b=np.copy(kw[2*l+1].T)
      nn.layers[l].b.shape=bs

  def set_weights(self, weights):
    nn=self.nn
    kw=weights
    for l in range(0,nn.nLayers):
      nn.layers[l].W=np.copy(kw[2*l])
      nn.layers[l].b=np.copy(kw[2*l+1])

  def get_weights(self):
    nn=self.nn
    kw=[]
    for l in range(0,nn.nLayers):
      kw.append( nn.layers[l].W )
      kw.append( nn.layers[l].b )
    return kw

  @staticmethod
  def __accuracy(y, yhat):
    ly=np.argmax(y, axis=0)
    lyhat=np.argmax(yhat, axis=0)
    return np.sum( ly==lyhat ) / len(ly)

  @staticmethod
  def __MSE(y, yhat):
    return np.mean( np.square(y-yhat) )

  def predict(self, x):
    return self.nn.doForward(x.T).T

  def evaluate(self, x, y):
    yhat = self.nn.predict(x.T)
    # self.nn.print()
    # input()
    loss = self.objective.doForward(yhat, y.T)
    metric = self.doMetric(y.T, yhat)
    return loss,metric

  def fit(self, x, y,  batch_size=32, shuffle=True, epochs=1, verbose=0):
    ''' run iteration until either tolerance is small enough or iterations
        is large enough.
        If batch_size is provided, then do minibatch of size batch_size.
        Return: loss and metric of the last batch of each epoch
    '''
    N=x.shape[0]
    if batch_size is None: batch_size = N

    Loss=[]

    for e in range(epochs):
      S=list(range(N))
      if shuffle:
        np.random.shuffle(S)
      for k in range(0, N, batch_size):
        I=S[k:k+batch_size]
        xb=x[I,:].T
        yb=y[I,:].T
        yhat = self.nn.doForward(xb)
        loss = self.objective.doForward(yhat, yb)
        dyhat= self.objective.doBackward(yb)
        self.nn.doBackward(dyhat)
        self.nn.updateWeights()
        b=k/batch_size
      if verbose==2:
        print('Epoch %d/%d: loss: %.4g' % (e, epochs, loss))
      Loss.append( loss )
    return Loss
