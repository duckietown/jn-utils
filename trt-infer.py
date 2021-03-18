import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import onnx2trt
TRT_LOGGER = trt.Logger()
# https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
class TRTSession:
    def __init__(self, engine):
        self.engine = engine
        print("a")
        self.context = engine.create_execution_context()
        # get sizes of input and output and allocate memory required for input data and for output data
        for binding in engine:
            if engine.binding_is_input(binding):  # we expect only one input
                input_shape = engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                self.device_input = cuda.mem_alloc(input_size)
            else:  # and one output
                self.output_shape = engine.get_binding_shape(binding)
                print(self.output_shape)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                self.host_output = cuda.pagelocked_empty(trt.volume(self.output_shape) * engine.max_batch_size, dtype=np.float32)
                self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        print("b")
    def infer(self, input):
        print("c")
        stream = cuda.Stream()
        host_input = np.array(input, dtype=np.float32, order='C')
        cuda.memcpy_htod_async(self.device_input, host_input, stream)
        self.context.execute_async(bindings=[int(self.device_input), int(self.device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, stream)
        stream.synchronize()
        print("d")
        reshape = list(self.output_shape)
        reshape.insert(0, -1)

        as_arr = np.array(self.host_output)
        print("e")
        return as_arr

if __name__ == "__main__":
    """    import onnx2trt
    import torch2onnx
    import torch

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layes = torch.nn.Sequential(torch.nn.Linear(200, 20), torch.nn.ReLU(), torch.nn.Linear(20,10), torch.nn.Sigmoid(), torch.nn.Linear(10, 1))

        def forward(self, input):
            return self.layes(input)
"""
    def numpy_input():
        return np.random.uniform(0,3, (1,640,640))
    """    def torch_input():
        x = torch.from_numpy(numpy_input())
        x = x.to(dtype=torch.float)
        return x
"""

    # Zeroth, just try the dummy model, just in case
   # dummy = Dummy()
   # print(dummy(torch_input()))

    # First, go from torch to onnx
   # torch2onnx.convert(Dummy(), torch_input(), torch.device("cpu"), "dummy.onnx")

    # Then, onnx to trt
#    engine = onnx2trt.main("yolov5s.onnx", 3, 1, 1, "dummy.engine")
    with open("engine.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    session = TRTSession(engine)
    print(session.infer(numpy_input()))
