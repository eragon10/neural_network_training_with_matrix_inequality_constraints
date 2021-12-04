

import network

import pandas as pd
import numpy as np
import tensorflow as tf


    
if __name__ == "__main__":
    
    
    # load trainings dataset from disk
    data = pd.read_csv("../data/data_pendulum.csv") 
    
    inputs =  np.transpose(
        data[['x','y']].to_numpy(dtype=np.float32)
    )
    outputs = np.transpose(
        data['u'].to_numpy(dtype=np.float32)
    )
    
    
    
    
    # define the LTI system 
    a = [[ 1.0000 , 0.0200 ],   
         [ 0.4000 , 0.9733 ]]
    
    b = [[0], [0.5333]]
    
    # define boundries
    h = [2.5,6.0]
    


    # create neural network
    nn = network.NeuralNetwork(4, system={'A': a, 'B': b, 'h': h}, opts={'phi': 12, 'act': tf.math.tanh} )


    nn.find(epochs=1600000, bound=-0.002)
    nn.save(filename='model_ipc_find_im.json')

    nn.fit(dataset={'x': inputs, 'y': outputs}, epochs=3000)
 
    # save computed weights/variables to disk
    nn.save(filename='model_ipc_final_im.json')
