import argparse

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import onnx
from onnxsim import simplify
import datetime
import time

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


# loosely inspired by https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
# https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#serial_model_python

def get_builder(ram_size, batch_size):
    """
    Makes a builder
    :param ram_size: in GB
    :param batch_size:
    :return:
    """
    builder = trt.Builder(TRT_LOGGER)
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = ram_size << 30
    # we have only one image in batch
    builder.max_batch_size =batch_size

    #if builder.platform_has_fast_int8: #what does int8 mode do? activating it for dummy test doesnt work
    #    builder.int8_mode = True
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    return builder

def onnx_to_trt(onnx_file, ram_size, batch_size, explicit_batch):
    builder = get_builder(ram_size, batch_size)

    # https://github.com/NVIDIA/TensorRT/issues/183#issuecomment-657673563
    # must have an explicit batch!
    explicit_batch = explicit_batch << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):  # https://forums.developer.nvidia.com/t/tensorrts-onnx-parser-cant-parse-the-output-layer-correctly/71252/9
            print(parser.get_error(0))
            raise IOError(parser.get_error(0))
    print('Completed parsing of ONNX file')

    # https://github.com/NVIDIA/TensorRT/issues/183
    last_layer = network.get_layer(network.num_layers - 1)
    # Check if last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))

    engine = builder.build_cuda_engine(network)

    print("Successfully buily engine")

    return engine

def serialize_engine(engine, outputfile):
    with open(outputfile, "wb") as f:
        f.write(engine.serialize())
    print("Successfully serialize & wrote engine")

def simplify_onnx(onnx_file):
    # https://github.com/daquexian/onnx-simplifier


    # load your predefined ONNX model
    model = onnx.load(onnx_file)
    # convert mode

    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    modelname = time.strftime("%d%m%Y%S")
    time.sleep(1)
    onnx.save(model_simp, modelname)
    return modelname

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer onnx model to TensorRT.')
    parser.add_argument('onnxfile', type=str, help='onnx file path (input)')
    parser.add_argument("enginefile", type=str, help="engine file path (output)")
    parser.add_argument('--ram', type=int, default=1, help='workspace size (in GB)')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--explicit_batch", type=int, default=1, help="explicit batch size")

    args = parser.parse_args()
    return args

def main(onnxfile, ram, batch_size, explicit_batch, enginefile):
    newname = onnxfile
    try:
        trt_engine = onnx_to_trt(newname, ram, batch_size, explicit_batch)
    except:
        newname = simplify_onnx(onnxfile)
        trt_engine = onnx_to_trt(newname, ram, batch_size, explicit_batch)
    serialize_engine(trt_engine, enginefile)
    return trt_engine


if __name__ == "__main__":
    args = parse_args()
    main(args.onnxfile, args.ram, args.batch_size, args.explicit_batch, args.enginefile)