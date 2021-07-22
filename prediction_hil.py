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

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')

    while True:
        state_recv = socket.recv().decode('utf-8')
        statejson = json.loads(state_recv)
        state = statejson["state"]
        action, _ = model.predict(state, greedy=True)
        json_action = json.dumps({'action': action.tolist()})
        socket.send(json_action.encode('utf_8'))

predict()
