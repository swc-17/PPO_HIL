import zmq
from ppo import PPO
import os
import gym
import numpy as np
import json
import tensorflow as tf
from tensorflow.python.framework import graph_util


def convert_to_pb():
    action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)
    input_shape = [67]
    model = PPO(input_shape, action_space,
                model_dir=os.path.join("models", 'pretrained_agent'))
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    print('------------convert to freeze model------------')
    graph_def = tf.get_default_graph().as_graph_def()
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        if 'output' in tensor_name:        
            print(tensor_name,'\n')
    
    output_graph_def = graph_util.convert_variables_to_constants(model.sess,graph_def,['policy/output'])
    model_f = tf.gfile.GFile("model.pb","wb")
    model_f.write(output_graph_def.SerializeToString())
    print('------------saved to model.pb------------')
   
convert_to_pb()
