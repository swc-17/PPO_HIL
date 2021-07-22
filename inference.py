import os
import zmq 
import json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n ' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()

class Agent(object):

    def __init__(self, file_path):
        self.engine = self.get_engine(file_path)
        self.allocate_buffs(self.engine)
      
    def get_engine(self, file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)         
        if os.path.exists(file_path):
            print("Reading engine from file {}".format(file_path))
            with open(file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print(file_path,'does not exist.')
            exit()

    def allocate_buffs(self, engine):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in engine:
            print(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)), dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def inference(self, data=None):
        self.inputs[0].host = data.ravel()
        with self.engine.create_execution_context() as context:
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
        results = [out.host for out in self.outputs]

        return results


def predict():

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')
    agent = Agent('model.trt')

    while True:
        state_recv = socket.recv().decode('utf-8')
        statejson = json.loads(state_recv)
        state = statejson["state"]
        # state = np.zeros((67),dtype=np.float32) + 0.5
        state = np.array(state,dtype=np.float32)
        # print(state.shape)
        action = agent.inference(state)[0]
        print(action)
        json_action = json.dumps({'action': action.tolist()})
        socket.send(json_action.encode('utf_8'))

import ipdb
#ipdb.set_trace()
predict()
