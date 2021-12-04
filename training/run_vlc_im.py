 




import network

import pandas as pd
import numpy as np
import tensorflow as tf




if __name__ == "__main__":
    
    
       
    # load trainings dataset from disk
    data = pd.read_csv("../data/data_vehicle.csv") 
    
    inputs =  np.transpose(
        data[['x','y','z','w']].to_numpy(dtype=np.float32)
    )
    outputs = np.transpose(
        data['u'].to_numpy(dtype=np.float32)
    )
    
    
    
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
    

    nn.find(epochs=1600000, bound=-0.002)
    nn.save(filename='test_model_vlc_find_im.json')

    nn.fit(dataset={'x': inputs, 'y': outputs}, epochs=3000)
 
    # save computed weights/variables to disk
    nn.save(filename='test_model_vlc_final_im.json')
    
