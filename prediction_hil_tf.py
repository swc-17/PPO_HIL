import zmq
from ppo import PPO
import os
import gym
import numpy as np
import json

def predict():
    action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)
    input_shape = [67]
    model = PPO(input_shape, action_space,
                model_dir=os.path.join("models", 'pretrained_agent'))
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    state = np.zeros((67),dtype=np.float32) + 0.5
    action, _ = model.predict(state, greedy=True)
    print(action) #[0.04324067 0.82321334]
predict()
