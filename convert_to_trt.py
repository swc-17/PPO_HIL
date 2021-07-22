import tensorrt as trt

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(uff_model_path, trt_logger, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, silent=False):
    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 30
        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        builder.max_batch_size = batch_size

        INPUT_NODE = 'input_state_placeholder'
        OUTPUT_NODE = 'policy/output'
        INPUT_SIZE = (1,67,1,1)
        
        parser.register_input(INPUT_NODE, INPUT_SIZE)
        parser.register_output(OUTPUT_NODE)
        parser.parse(uff_model_path, network)

        if not silent:
            print("Building TensorRT engine. This may take few minutes.")

        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def convert_to_trt():
   
    print('---------------convert to trt--------------------')
    INPUT_NODE = 'input_state_placeholder'
    OUTPUT_NODE = 'policy/add'
    MAX_BATCH_SIZE = 1 
    MAX_WORKSPACE = 1<<30
    Logger = trt.Logger(trt.Logger.ERROR)
    engine = build_engine('model.uff', Logger)
    save_engine(engine, 'model.trt')
    print('---------------saved to model.trt--------------------')

convert_to_trt()
