import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import cv2
import numpy as np
import torch

from utils.decode import decode
from utils.post_process import post_process
import time 

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
HEIGHT_RESIZE = 1088
WIDTH_RESIZE  = 1440

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n ' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()

class Detector(object):

    def __init__(self, filepath, calib):
        self.engine = self.get_engine(filepath)
        self.allocate_buffs(self.engine)
        self.class_names = ['Pedestrian', 'Car', 'Cyclist']
        self.calib = calib

    def get_engine(self, filepath):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        filename = filepath.split(".")[0]
        suffix = filepath.split(".")[1]
        if suffix == "onnx":
            onnx_file_path = filepath
            engine_file_path = filename + ".trt"
        else:
            onnx_file_path = filename + ".onnx"
            engine_file_path = filepath

        def build_engine():
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30 # 1GB
                builder.max_batch_size = 1
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    parser.parse(model.read())
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine
                
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

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

    def inference(self, data):
        data = self.pre_process(data)
        self.inputs[0].host = data.ravel()
        # s = time.time()

        with self.engine.create_execution_context() as context:
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
        # e = time.time()
        # print('time: ',e - s)
        output = [out.host for out in self.outputs]
        dets = self.post_process(output)

        return dets

    def pre_process(self, data):
        data = data.astype(np.float32) / 255.
        data = (data - MEAN) / STD
        data = data.transpose(2, 0, 1)
        
        return data

    def post_process(self, h_output):
        output = {}
        output["hm"] = torch.from_numpy(h_output[0]).reshape(3, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0)
        output["dep"] = torch.from_numpy(h_output[1]).reshape(1, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0)
        output["rot"] = torch.from_numpy(h_output[2]).reshape(8, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0)
        output["dim"] = torch.from_numpy(h_output[3]).reshape(3, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0) 
        output["wh"] = torch.from_numpy(h_output[4]).reshape(2, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0) 
        output["reg"] = torch.from_numpy(h_output[5]).reshape(2, int(HEIGHT_RESIZE / 4), int(WIDTH_RESIZE / 4)).unsqueeze(0) 
        output['hm'] = output['hm'].sigmoid_()
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        
        dets = decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], output['wh'], output['reg'])
        dets = dets.detach().cpu().numpy()
        results = post_process(dets, self.calib)
        return results

        
