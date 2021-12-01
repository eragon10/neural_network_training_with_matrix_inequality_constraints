 




import network


import numpy as np
import tensorflow as tf




if __name__ == "__main__":
    
    # define the LTI system 
    a = [[  1.0000, 0.0200, 0,      0       ],
           [  0,      0.9027, 2.7231, 0.0236  ],
           [  0,      0,      1.0000, 0.0200  ],
           [  0,      0.0188, -0.5254,0.8565  ]]
    
    b = [[0], [1.4753], [0], [1.1615]]
    
    # define boundries
    h = [2.0,5.0,1.0,5.0]
    


    # create neural network
    nn = network.NeuralNetwork(4, system={'A': a, 'B': b, 'h': h}, opts={'phi': 12, 'act': tf.math.tanh} )



    inital_values = np.dot( np.diag(h), np.random.uniform(-1.0,1.0, size=(4,100)) ).tolist()

    nn.find(epochs=1600000, bound=-0.002)
    nn.save(filename='model_vlc_find.json')

    nn.fit(dataset={'x0': inital_values, 'N': 120}, epochs=3000)
 
    # save computed weights/variables to disk
    nn.save(filename='model_vlc_final.json')
    
