import ctypes
import numpy as np
import os

# Load the shared library
lib_path = os.path.abspath("unet/libunet.so")
lib = ctypes.CDLL(lib_path)

# Set argument and return types for ConvReLU functions
lib.create_conv_relu.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.create_conv_relu.restype = ctypes.c_void_p

lib.destroy_conv_relu.argtypes = [ctypes.c_void_p]
lib.destroy_conv_relu.restype = None

lib.conv_relu_forward.argtypes = [
    ctypes.c_void_p,               # ConvReLU*
    ctypes.POINTER(ctypes.c_float),  # input float*
    ctypes.POINTER(ctypes.c_float),  # output float*
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int  # N, C, H, W
]
lib.conv_relu_forward.restype = None

lib.conv_relu_backward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.conv_relu_backward.restype = None


class ConvReLU:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=1):
        self.obj = lib.create_conv_relu(in_channels, out_channels, filter_size)
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        if not self.obj:
            raise RuntimeError("Failed to create ConvReLU object")

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        assert input_tensor.ndim == 4
        N, C, H, W = input_tensor.shape
        input_tensor = input_tensor.astype(np.float32, copy=False)

        # Allocate output tensor (you may want to compute size based on padding/stride)
        out_H = int((H + 2*self.padding - self.filter_size) / self.stride + 1)  # assuming kernel size = 3 and stride=1
        print("Output height:", out_H)
        out_W = int((H + 2*self.padding - self.filter_size) / self.stride + 1)
        output_tensor = np.empty((N, self.out_channels, out_H, out_W), dtype=np.float32)

        lib.conv_relu_forward(
            self.obj,
            input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, C, H, W
        )

        return output_tensor

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert grad_output.ndim == 4
        N, C, H, W = grad_output.shape
        grad_output = grad_output.astype(np.float32, copy=False)
        grad_input = np.empty((N, C, H + 2, W + 2), dtype=np.float32)  # adjust dims if needed

        lib.conv_relu_backward(
            self.obj,
            grad_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, C, H, W
        )

        return grad_input

    def __del__(self):
        if self.obj:
            lib.destroy_conv_relu(self.obj)
            self.obj = None

    @property
    def out_channels(self):
        # You'll need to hardcode or store this somewhere
        # since there's no getter for `output_channels` in C++
        return 64  # example — match what you used in constructor


if __name__ == "__main__":
    conv = ConvReLU(in_channels=3, out_channels=64, filter_size=3)

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = conv.forward(x)

    print("Output shape:", y.shape)

    dy = np.random.randn(*y.shape).astype(np.float32)
    dx = conv.backward(dy)

    print("Gradient shape:", dx.shape)
