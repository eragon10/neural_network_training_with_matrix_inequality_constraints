
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import pandas as pd 
import numpy as np
        
        
        

    
   
def block_diagonal(matrices, dtype=tf.float32):
    
    dims  = [ m.shape[1] for m in matrices ]
        
    arrayitem = [
        tf.pad( matrices[i] , [[0,0],[ sum(dims[0:i]), sum(dims[i+1:]) ]]) for i in range(len(dims))
    ]
    
    return tf.concat(arrayitem,0)




class NeuralNetwork:
    
    
    def save(self, filename='controller.json'):
    
        P, W = self.init()
        
        weights = [ np.array( w.numpy() ,dtype=np.float64 ) for w in W ]
        lambdas = [ np.array( w.numpy(), dtype=np.float64 ) for w in self.lambdas ]
        lyapuno = np.array( P.numpy() , dtype=np.float64 )
        alphas  = [ np.array( w.numpy(), dtype=np.float64 ) for w in self.alphas ]
        
        datacontainer = {
            'weights': [ {'rows': w.shape[0] , 'cols': w.shape[1], 'data':
                            list(1.0*np.reshape(w,(w.shape[0]*w.shape[1]))) } for w in weights ],
            'lambdas': [ {'size': w.shape[0], 'data': list(1.0*w) } for w in lambdas ],
            'alphas': [ {'size': w.shape[0], 'data': list(1.0*w) } for w in alphas ],
            'lyapu' : { 'rows': lyapuno.shape[0], 'cols': lyapuno.shape[1], 
                        'data': list(1.0*np.reshape(lyapuno,(lyapuno.shape[0]*lyapuno.shape[1]))) }
        }
        
        with open(filename, "w") as outfile:
            json.dump(datacontainer, outfile)
        


    def __init__( self, nlayer, system, opts ):
        '''
            This function initilizes all relevant variables and 
            constants for feasibility searching and training
            
            :param dims:    describe the network structure as array of neurons per layer
            :param system:  decribe the LTI system x_dot = A x + B u with boundries h 
                            as dictionary in the form {'A': a, 'B': b, 'h': h}
            :param opts:    describe additional settings for training. Primary use it to
                            set the activation function.
        '''
        
        
        self.nu = 1.0
        
        # prepare the system matrices for training:
        #
        #   x_dot = A x + B u
        #
        self.A = tf.constant( system['A'] )
        self.B = tf.constant( system['B'] )
        
        # store the dimensions of the neural 
        # network 
        self.dims = [
            int(tf.shape(self.B)[0]) for i in range(nlayer-1)
        ] + [ int(tf.shape(self.B)[1]) ] 
        
        
        
        # prepare constants for LMI of the format:
        #
        #   P - H_i^T h_i^{-2} H_i > 0
        #
        self.h = [
            tf.constant( (v*v) , dtype=tf.float32, shape=[1,1] ) for v in system['h']
        ]
        
        # prepare constant for the first inequality constraint:
        #
        #   vhead_1 - |W_1| G > 0
        #
        self.G = tf.constant( system['h'], dtype=tf.float32, shape=[len(system['h'])] )
    
        
        # prepare alphas for training
        self.alphas = [
            tf.Variable( [0.99 for i in range(self.dims[i])], dtype=tf.float32, 
                        name=('alpha{}').format(i) ) for i in range(1,len(self.dims)-1)
        ]
            
        self.temporaries = [
            tf.Variable( tf.zeros([self.dims[i],self.dims[i]]), name=('temporary{}').format(i) ) for i in range(0,len(self.dims)-2)
        ]
        

        # define the used activation function
        self.activation = opts['act'] 
        
        
        # create the variable phi for feasibility searching for
        # usage in the barrier function:
        #
        #   -logdet( phi * I - R_v^T Z R_v - Rphi^T M Rphi )
        #
        self.phi = tf.Variable( opts['phi'], dtype=tf.float32, name=("phi") )
        
        
        self.pweights = [
            tf.Variable( tf.random.normal([self.dims[i+1], self.dims[i]], stddev=0.1), name=('X{}').format(i) )
                    for i in range(len(self.dims)-2)
        ] + [ 
            tf.Variable( tf.random.normal([self.dims[-1],self.dims[-2]], stddev=0.1), name=('W{}').format(len(self.dims)-2) )
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
            
            
        # create random matrix for initilization of Lyapunov matrix P
        seed = tf.random.normal([ self.dims[0], self.dims[0]], stddev=0.1)
        
        # create Lyapunov variable P with symmetric and positive definite
        # initial value
        self.lyapu = tf.Variable(
            seed + tf.transpose(seed) + 1*tf.eye( self.dims[0] ) , name=('P')
        )
        

        self.Q = tf.constant( np.eye(self.dims[0]) , dtype=tf.float32 )
        self.R = tf.constant( 0.001*np.eye(self.dims[-1]) , dtype=tf.float32 )
        self.S = tf.constant( 20.0*np.eye(self.dims[0]) , dtype=tf.float32 )
        
        
        
        # use ADAM optimizer for training
        self.optimizer = tf.keras.optimizers.Adam( learning_rate=0.0005, beta_1=0.9, beta_2=0.9999,
                                                   epsilon=1e-07, amsgrad=False)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        #self.optimizer = tf.keras.optimizers.Adagrad(
        #    learning_rate=0.001, initial_accumulator_value=0.2, epsilon=1e-07)
        
  
  
        # use gradient decent for feasibility searching
        self.foptimizer = tf.keras.optimizers.Adam( learning_rate=0.0005, beta_1=0.9, beta_2=0.9999,
                                                   epsilon=1e-07, amsgrad=False)
        #self.foptimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)

        # middleware code for debug printing
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')


    
    @tf.function
    def boundpropagation(self, W):
        
        r = self.G
        vheads = []
        for i in range(len(self.dims)-2):
            v = tf.linalg.matvec( tf.math.abs( W[i]), r)
            r = self.activation(v)
            vheads = vheads + [v]
        
        alphas = [
            tf.math.divide( self.activation( vheads[i] ) , vheads[i] ) for i in range(len(vheads))
        ]
        
        for i in range(len(alphas)):
            self.alphas[i].assign( alphas[i] )
            
            
    
        temps = [
            2.0 * tf.matmul( tf.matmul( self.pweights[i], tf.matmul( tf.linalg.diag(self.alphas[i]), tf.linalg.diag(self.lambdas[i]) ), transpose_a=True ), self.pweights[i] ) for i in range(0,len(self.dims)-2)
        ]
            
        for i in range(len(temps)):
            self.temporaries[i].assign( temps[i] )
            
            
        #self.temporaries[0].assign(
        #    2.0 * tf.matmul( tf.matmul( self.pweights[0][1], tf.slice(tf.matmul( tf.linalg.diag(self.alphas[0]), tf.linalg.diag(self.lambdas[0]) ),[2,2],[6,6]) , transpose_a=True ), self.pweights[0][1] )
        #  + 2.0 * tf.matmul( tf.matmul( self.pweights[0][0], tf.slice(tf.matmul( tf.linalg.diag(self.alphas[0]), tf.linalg.diag(self.lambdas[0]) ),[0,0],[2,2]) , transpose_a=True ), self.pweights[0][0] )
        #)
        
        
        #self.temporaries[1].assign(
        #    2.0 * tf.matmul( tf.matmul( self.pweights[1][0], tf.matmul( tf.linalg.diag(self.alphas[1]), tf.linalg.diag(self.lambdas[1]) ) , transpose_a=True ), self.pweights[1][0] )
        #)


    @tf.function
    def init(self):
        '''
            This function computes P = 0.5*(P_old + P_old^T) to ensure that
            during training P will always be symmetric. It returns P for
            usage in the computation graph.
        '''
        
        #W = [
        #    tf.concat([nu*tf.eye(2) + self.pweights[0][0], self.pweights[0][1]], axis=0 )
        #] + [
        #    nu*tf.eye(8) + self.pweights[1][0]
        #] + [
        #    self.pweights[2][0]
        #]
        
        W = [
            self.nu*tf.eye(tf.shape(self.pweights[i])[0]) + self.pweights[i] for i in range(len(self.pweights)-1)
        ] + [
            self.pweights[-1]
        ]
        

         
        self.boundpropagation(W)
        
        P = tf.constant(0.5, dtype=tf.float32) * ( self.lyapu + tf.transpose(self.lyapu) )
        
        return P, W


    @tf.function
    def query(self, x, W):
        '''
            This function is the query/evaluation function of
            the neural network. It just computes:
            
                x_0 = x                 for all i = 0,1,...,L
                z_i = W_i x_i
                x_{i+1} = activation(z_i)
                u = z_L
                
                        
            :param x: decribes the input of the neural network
            
        '''
        
        for i in range(len(W)-1):
            x = self.activation( tf.matmul( W[i],x) )
            
        return tf.matmul( W[-1],x)
    
    
    @tf.function
    def regionofatraction(self, P):
        '''
            This function generates the LMI's of the format:
            
                --                 --
                |   h_i^2    H_i^T  |
                |   H_i      P      |       > 0
                --                 --
                
            which is equivalent to (using the Schur complement):
            
                 P - H_i^T h_i^{-2} H_i > 0
            
            Here H is hardcoded to H = I the identity matrix for
            convenience. It returns a list of all necessary
            matrices of this form.
            
            :param P: describes the Lyapunov matrix P
        '''
        Hi = [
            tf.concat([ tf.zeros([i,1]), tf.constant(1.0,dtype=tf.float32,shape=[1,1]), tf.zeros([self.dims[0]-i-1,1]) ], axis=0 )  for i in range(self.dims[0])
        ]
        
        return [
             tf.concat( [ tf.concat([ self.h[i],  tf.transpose(Hi[i]) ], axis=1 ), tf.concat([ Hi[i], P ], axis=1 ) ], axis=0 )
                for i in range(self.dims[0])
        ]
    
        
    
    
    
    @tf.function
    def stability(self, P, W):


        T = [
            tf.linalg.diag(lam) for lam in self.lambdas
        ]

        ditems = [ P ] + [ 2*T[i] for i in range(len(T)) ] + [ P ]
        
        
        
        #l0 = nu*tf.slice( tf.matmul(tf.linalg.diag(2.0*self.alphas[0]), T[0]), [0,0], [2,2] )
        #l1 = nu*tf.matmul(tf.linalg.diag(2.0*self.alphas[1]), T[1])
        
        middle = [
            self.nu*tf.matmul(tf.linalg.diag(2.0*self.alphas[i]), T[i]) for i in range(len(self.dims)-2)
        ]
        
        diag_items = [
            ditems[i] + self.nu*middle[i] + tf.matmul(middle[i], self.pweights[i]) + tf.matmul(self.pweights[i], middle[i], transpose_a=True) + self.temporaries[i] for i in range(len(self.dims)-2)
        ] + [ ditems[-2] ] + [ ditems[-1] ]


        subdiag_items = [
           tf.matmul( tf.linalg.diag(self.alphas[i]+1.0),  tf.matmul( T[i], W[i] )) for i in range(len(T))
        ] + [ -1.0*tf.matmul( P, tf.matmul( self.B, W[-1] ) ) ]

        dpart = tf.linalg.LinearOperatorBlockDiag(
             [ tf.linalg.LinearOperatorFullMatrix(m) for m in diag_items ] ).to_dense()

        spart = tf.pad( tf.linalg.LinearOperatorBlockDiag(
             [ tf.linalg.LinearOperatorFullMatrix(m) for m in subdiag_items ] ).to_dense(),
                    [[self.dims[0],0],[0,self.dims[0]]] )

        opart = tf.pad( tf.matmul( self.A, P, transpose_a=True ),
                       [[0, sum(self.dims[1:-1])+self.dims[0] ], [ sum(self.dims[0:-1]) ,0 ]])

        return dpart - spart + opart - tf.transpose(spart) + tf.transpose(opart)



    

    
    @tf.function
    def step_feasibility(self, rho = 0.01):
        '''
            This function describes the update step during each iteration 
            in the feasibility searching process. Here all constraints come
            together and will be applied to their barrier functions.
            
            We move forward first, then calculate gradients to move backwards.
            
            :param rho: value describing the value in front of the barrier
                        functions
        '''
        
        with tf.GradientTape() as tape:
            
            # first get P to use in training
            P, W = self.init()
            
            # get stability matrix Y
            Y = self.stability(P, W)  
            
            # get all other constraints (bounds, ...) (disabled at the moment)
            c = self.regionofatraction(P)
            
            # apply the barrier function -logdet(phi*I + Y) and the
            # objective 5*phi in order to minimize phi
            o = tf.constant(1.0, dtype=tf.float32) * self.phi - rho * tf.linalg.logdet( 
                Y + self.phi * tf.eye(tf.shape(Y)[0]) )
             
            
            # add all other constraints to the objective function with
            # their individual barrier functions (disabled at the moment)
            for mat in c:
                Mtilde = mat + self.phi * tf.eye( tf.shape(mat)[0] )
                o = o - rho * tf.linalg.logdet( Mtilde )
            



            variables = [self.phi, self.lyapu] + self.pweights + self.lambdas
            gradients = tape.gradient(o,variables)
            
            # update the variables using the gradient decent optimizer
            self.foptimizer.apply_gradients(zip(gradients, variables))
            
            return o




    @tf.function
    def trajectories(self, W, inital_values, N=100, gamma=0.92):
        x = tf.constant( inital_values, dtype=tf.float32 )
        s = tf.shape( x )

        loss = tf.zeros( [s[1]] )
        for k in range(N):
            u = self.query(x, W)
            x =  tf.matmul( self.A,  x ) +  tf.matmul( self.B, u )

            loss = loss + gamma**(N-k-1) * (  tf.math.reduce_sum( tf.math.multiply( x,  tf.matmul( self.Q, x ) ), axis=0 )
                                        +   tf.math.reduce_sum( tf.math.multiply( u,  tf.matmul( self.R, u ) ), axis=0 )      )

        objective = tf.math.divide( loss, tf.constant( N, dtype=tf.float32 ) ) +  tf.math.reduce_sum(
                                tf.math.multiply( x,  tf.matmul( self.S, x ) ), axis=0 )

        return tf.math.reduce_sum( objective ) / tf.cast( tf.shape(objective)[0] , dtype=tf.float32 ) / 10.0

            
    @tf.function
    def step(self, x0, N, rho = 0.01):
        '''
            This function describes the update step during each iteration 
            in the training process. Here all constraints come
            together and will be applied to their barrier functions.
            
            We move forward first, then calculate gradients to move backwards.
            
            :param rho: value describing the value in front of the barrier
                        functions
            :param x:   training dataset, e.g. input dataset
            :param y:   training dataset, e.g. target dataset
        '''
        
        with tf.GradientTape() as tape:
            
            P, W = self.init()
            Y = self.stability(P, W)
            
            c = self.regionofatraction(P)

            l =  self.trajectories( W, x0, N )
            
            # apply the loss function (squared error) with a weight
            #o = tf.constant(2.0, dtype=tf.float32) * tf.square( r - y )
            o =  l - rho * tf.linalg.logdet(Y)
            for mat in c:
                o = o - rho * tf.linalg.logdet( mat )


            
            # tell Tensorflow which variables should be updated
            variables = self.pweights + [self.lyapu] + self.lambdas
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
        
        
        for rho in [0.01, 0.008]:
            for epoch in range(epochs):
                # execute training step
                self.step( dataset['x0'], dataset['N'], rho=rho )
            
                if epoch % 1000 == 0:
                    print( template.format( epoch, self.train_loss.result() ) )
                
                self.train_loss.reset_states()
            
        
    def find(self, epochs, bound=-0.02, rho=0.01 ):
        '''
        This find function runs feasibility searching.
        
        :param nn:         decribes the neural network
        :param epochs:     decribes the maximum amount of iterations
        '''
        template = 'Iteration: {}, Phi: {}, Obj: {}'
        for epoch in range(epochs):

            # execute update step
            obj = self.step_feasibility( rho=rho )
            
            
            # check if it is feasible and stop searching 
            # if condition is satisfied 
            if self.phi < tf.constant(bound, dtype=tf.float32):
                break
            
            # just some debug printing of phi and the objective function
            if epoch % 1000 == 0:
                print( template.format(epoch, float(self.phi), float(obj) ) )
        
        
    
    

    
if __name__ == "__main__":
    
    # define the LTI system 
    a = [[ 1.0000 , 0.0200 ],   
         [ 0.4000 , 0.9733 ]]
    
    b = [[0], [0.5333]]
    
    # define boundries
    h = [2.5,6.0]
    


    # create neural network
    nn = NeuralNetwork(4, system={'A': a, 'B': b, 'h': h}, opts={'phi': 12, 'act': tf.math.tanh} )


    inital_values = np.dot( np.diag(h), np.random.uniform(-1.0,1.0, size=(2,100)) ).tolist()


    nn.find(epochs=1600000, bound=-0.002)
    nn.save(filename='model_ipc_find.json')

    nn.fit(dataset={'x0': inital_values, 'N': 120}, epochs=3000)
 
    # save computed weights/variables to disk
    nn.save(filename='model_ipc_final.json')
    




    

