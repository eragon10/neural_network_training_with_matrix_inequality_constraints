
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import pandas as pd 
import numpy as np
        
        
        
import matplotlib.pyplot as plt
    
   
def block_diagonal(matrices, dtype=tf.float32):
    
    dims  = [ m.shape[1] for m in matrices ]
        
    arrayitem = [
        tf.pad( matrices[i] , [[0,0],[ sum(dims[0:i]), sum(dims[i+1:]) ]]) for i in range(len(dims))
    ]
    
    return tf.concat(arrayitem,0)




class NeuralNetwork:
    
    
    def save(self, filename='controller.json'):
    
        weights = [ np.array( w.numpy() ,dtype=np.float64 ) for w in self.weights ]
        biases =  [ np.array( w.numpy() ,dtype=np.float64 ) for w in self.biases ]
        lambdas = [ np.array( w.numpy(), dtype=np.float64 ) for w in self.lambdas ]


        datacontainer = {
            'weights': [ {'rows': w.shape[0] , 'cols': w.shape[1], 'data':
                            list(1.0*np.reshape(w,(w.shape[0]*w.shape[1]))) } for w in weights ],
            'biases': [ {'size': w.shape[0] , 'data': list(1.0*np.reshape(w,(w.shape[0]*w.shape[1]))) } for w in biases ],
            'lambdas': [ {'size': w.shape[0], 'data': list(1.0*w) } for w in lambdas ],
        }
        
        with open(filename, "w") as outfile:
            json.dump(datacontainer, outfile)
        


    def __init__( self, dims, opts ):
        '''
            This function initilizes all relevant variables and 
            constants for feasibility searching and training
            
            :param dims:    describe the network structure as array of neurons per layer
            :param system:  decribe the LTI system x_dot = A x + B u with boundries h 
                            as dictionary in the form {'A': a, 'B': b, 'h': h}
            :param opts:    describe additional settings for training. Primary use it to
                            set the activation function.
        '''

        
        # store the dimensions of the neural 
        # network 
        self.dims = dims
        
        

        # define the used activation function
        self.activation = opts['act'] 
        self.clipschitz  = opts['lipschitz']
        
        
        self.weights = [
            tf.Variable( tf.random.normal([self.dims[i+1], self.dims[i]], stddev=0.1), name=('W{}').format(i) )
                    for i in range(len(self.dims)-1)
        ]
        
        self.biases = [
            tf.Variable( tf.random.normal([self.dims[i+1], 1], stddev=0.1), name=('B{}').format(i) )
                    for i in range(len(self.dims)-1)
        ]
    
        
        # create lambdas in parts as vectors for the matrix M:
        #
        #   --                                          --
        #   | -2*alpha*beta*lambda   (alpha+beta)*lambda |
        #   |  (alpha+beta)*lambda     -2*lambda         |
        #   --                                          --
        #
        self.lambdas = [
            tf.Variable( tf.math.abs(tf.random.normal( [self.dims[i]], stddev=0.1, mean=10.0)), name=('lambda{}').format(i) )
                 for i in range(1,len(self.dims)-1)
        ]
            

        # use ADAM optimizer for training
        self.optimizer = tf.keras.optimizers.Adam( learning_rate=0.001, beta_1=0.94, beta_2=0.999,
                                                   epsilon=1e-07, amsgrad=False)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        #self.optimizer = tf.keras.optimizers.Adagrad(
        #    learning_rate=0.001, initial_accumulator_value=0.2, epsilon=1e-07)
        

        # middleware code for debug printing
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        
        
        
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, axis=0)





    @tf.function
    def query(self, x):
        '''
            This function is the query/evaluation function of
            the neural network. It just computes:
            
                x_0 = x                 for all i = 0,1,...,L
                z_i = W_i x_i
                x_{i+1} = activation(z_i)
                u = z_L
                
                        
            :param x: decribes the input of the neural network
            
        '''
        
        for i in range(len(self.weights)-1):
            x = self.activation( tf.matmul( self.weights[i],x) + self.biases[i] )
            
        return tf.matmul(self.weights[-1],x) + self.biases[-1]
    
    
    
    @tf.function
    def lipschitz(self):


        T = [
            tf.linalg.diag(lam) for lam in self.lambdas
        ]
        

        diag_items = [ self.clipschitz**2*tf.eye(self.dims[0]) ] + [ 2*T[i] for i in range(len(T)) ] + [ tf.eye(self.dims[-1]) ]
    
    

        subdiag_items = [
           tf.matmul( T[i], self.weights[i] ) for i in range(len(T))
        ] + [  self.weights[-1] ]

        dpart = tf.linalg.LinearOperatorBlockDiag(
             [ tf.linalg.LinearOperatorFullMatrix(m) for m in diag_items ] ).to_dense()

        spart = tf.pad( tf.linalg.LinearOperatorBlockDiag(
             [ tf.linalg.LinearOperatorFullMatrix(m) for m in subdiag_items ] ).to_dense(),
                    [[self.dims[0],0],[0,self.dims[-1]]] )
        
        return dpart - spart - tf.transpose(spart)



    

    
    @tf.function
    def step(self, x, y, rho = 0.01):

        
        with tf.GradientTape() as tape:
    
            
            Y = self.lipschitz()
            
       
            #l = tf.cast(self.loss(y, tf.nn.softmax(self.query(x),axis=0)), tf.float32) / x.shape[1]
            
            l = 10*self.cce(y,self.query(x)) #tf.nn.softmax(res,axis=0))
            #l = tf.math.reduce_sum(
            #            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=res, axis=0)) / tf.constant(x.shape[1], tf.float32)

            
            
            o =  l - rho * tf.linalg.logdet(Y)
        
            #o = l

            
            # tell Tensorflow which variables should be updated
            variables = self.weights + self.biases + self.lambdas 
            gradients = tape.gradient(o, variables)
            
            # update the variables using the gradient decent optimizer
            self.optimizer.apply_gradients(zip(gradients, variables))

            # middleware execution for debug printing
            self.train_loss(l)

            return l
      
      
              
    def fit(self, dataset, epochs):
        '''
        This fit function runs training.
        
        :param nn:         decribes the neural network
        :param dataset:    decribes the trainings dataset
        :param epochs:     decribes the amount of iterations
        '''
        template = 'Iteration: {}, Loss: {}'
        
        batch = 2000
        epoch = 0
        
        for rho in [0.01]:
            while epoch < epochs:
                # execute training step
                
                for i in range(int(dataset['x'].shape[1] / batch)):
                    self.step( dataset['x'][:,i*batch:(i+1)*batch], dataset['y'][:,i*batch:(i+1)*batch], rho=rho )
                    
                    if epoch % 1000 == 0:
                        print( template.format( epoch, self.train_loss.result() ) )
                        
                    self.train_loss.reset_states()
                    
                    epoch = epoch + 1
            
        
    
    

    
if __name__ == "__main__":
    
    """
    # load trainings dataset from disk
    data = pd.read_csv("sample_training_data.csv")

    inputs =  np.transpose(
        data[['x','y']].to_numpy(dtype=np.float32)
    )
    
    outputs = np.zeros([3,inputs.shape[1]])
    
    idx = np.transpose(
        data['class'].to_numpy(dtype=np.int8)
    )
    


    outputs[0,idx == 0] = 1
    outputs[1,idx == 1] = 1
    outputs[2,idx == 2] = 1
    """
    
    f = open('mnist_training.json')
    data = json.load(f)
    
    inputs = np.zeros((data['mnist']['size'][1], data['mnist']['size'][0]), dtype=np.float32)
    ouputs = np.zeros((data['mnist']['size'][2], data['mnist']['size'][0]), dtype=np.float32)
    
    for i in range(data['mnist']['size'][0]):
        inputs[:,i] = data['mnist']['data'][i]['x']
        ouputs[:,i] = data['mnist']['data'][i]['y']
    

  
    #plt.scatter(inputs[0,idx == 0],inputs[1,idx == 0] )
    #plt.scatter(inputs[0,idx == 1],inputs[1,idx == 1] )
    #plt.scatter(inputs[0,idx == 2],inputs[1,idx == 2] )
    #plt.show()

    # create neural network
    nn = NeuralNetwork([196,100,30,10], opts={'lipschitz': 20, 'act': tf.math.tanh} )
    nn.fit(dataset={'x': inputs, 'y': ouputs}, epochs=12000)
    
    
    nn.save(filename='model_mnist_lip.json')
    
    """ 
    # load trainings dataset from disk
    data = pd.read_csv("sample_testing_data.csv")

    inputs =  np.transpose(
        data[['x','y']].to_numpy(dtype=np.float32)
    )
    
    outputs = np.zeros([3,inputs.shape[1]])
    
    idx = np.transpose(
        data['class'].to_numpy(dtype=np.int8)
    )
    
    #print(outputs)
    res = tf.nn.softmax(nn.query(inputs),axis=0)
    
    print(idx == np.argmax(res, axis=0))
    print('accuracy', np.sum( idx == np.argmax(res, axis=0) ) / outputs.shape[1] )
 
    # save computed weights/variables to disk
    nn.save(filename='model_lip.json')
    """



    

