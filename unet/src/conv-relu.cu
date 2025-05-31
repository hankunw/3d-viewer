#include "../include/conv-relu.h"

#include <cassert>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>


/* Checking if the access is within original image bounds. 
 * If not zero is returned (equivalent with padding original input iamge with zeros */
template<int filter_size>
__device__
static inline float access_padded(Tensor <float, 4>& tensor, int b, int c, int x, int y, int w, int h) {
	x -= filter_size / 2;
	y -= filter_size / 2;
	if (0 <= x && x < w && 0 <= y && y < h)
		return tensor(b, c, x, y);
	return 0.;
}


template<int filter_size>
__global__
static void convolution_relu_forward(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = input.dim(0),
		input_channels = input.dim(1),
		output_channels = output.dim(1),
		width = output.dim(2),
		height = output.dim(3);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		float val = channel_bias;

		// Convolution:
		for (int c_in = 0; c_in < input_channels; c_in++) {
			for (int i = 0; i < filter_size; i++) {
				for (int j = 0; j < filter_size; j++) {
					float input_val = access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height);
					val += input_val * weights(c_out, c_in, i, j);
				}
			}
		}

		// ReLU:
		output(b, c_out, x, y) = fmaxf(0., val);
	}
}

__global__
static void relu_backward(Tensor<float, 4> error, Tensor<float, 4> output)
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error.dim(0), channels = error.dim(1), width = error.dim(2), height = error.dim(3);
	if (c >= channels || x >= width || y >= height)
		return;

	for (int b = 0; b < batch_size; b++) {
		float buf[2];
		buf[0] = 0.;
		buf[1] = error(b, c, x, y);
		error(b, c, x, y) = buf[output(b, c, x, y) != 0.];
	}
}

template<int filter_size>
__global__
static void convolution_backward(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float, 4> weights) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c_in = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = error.dim(0),
		input_channels = next_error.dim(1),
		output_channels = error.dim(1),
		width = next_error.dim(2),
		height = next_error.dim(3);

	if (x >= width || y >= height || c_in >= input_channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {
			for (int i = 0; i < filter_size; i++) {
				for (int j = 0; j < filter_size; j++) {
					float error_val = access_padded<filter_size>(error, b, c_out, x + i, y + j, width, height);
					val += error_val * weights(c_out, c_in, filter_size - i - 1, filter_size - j - 1);
				}
			}
		}

		next_error(b, c_in, x, y) = val;
	}
}


template<int filter_size>
__global__
static void convolution_gradient_weights_reduction(
	Tensor<float, 4> input, Tensor<float, 4> error,
	Tensor<float, 4> gradient_weights) {
	// Let's hope this kernel is memory bound (it surly is) or the compiler optimizes the divisions.
	const int x = (blockIdx.y / filter_size) * blockDim.y * 2 + threadIdx.y;
	const int y = (blockIdx.x / filter_size) * blockDim.x * 2 + threadIdx.x;
	const int i = (blockIdx.y % filter_size), j = (blockIdx.x % filter_size);

	const int batch_size = input.dim(0),
		input_channels = input.dim(1),
		output_channels = error.dim(1),
		width = error.dim(2),
		height = error.dim(3);

	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int c_out = z / input_channels, c_in = z % input_channels;

	const int threads_in_block = blockDim.x * blockDim.y;
	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	assert(c_in < input_channels && c_out < output_channels && x < width && y < height);

	// In U-nets always true
	bool val2inbound = (x + blockDim.y < width && y + blockDim.x < height);

	extern __shared__ float sm[];

	float val = 0.;
	for (int b = 0; b < batch_size; b++) {
		val += access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height) * error(b, c_out, x, y);
		if (val2inbound)
			val += access_padded<filter_size>(input, b, c_in, x + i + blockDim.y, y + j + blockDim.x, width, height) * error(b, c_out, x + blockDim.y, y + blockDim.x);
	}
	sm[tid] = val;

	__syncthreads();

	int n = threads_in_block;
	while (n > 1) {
		n /= 2;
		if (tid < n)
			sm[tid] += sm[tid + n];

		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(&gradient_weights(c_out, c_in, i, j), sm[tid]);
}


/* Similar concept as convolution_gradient_weights_reduction */
__global__
static void convolution_gradient_bias_reduction(Tensor<float, 4> error, Tensor<float, 1> gradient_bias) {
	const int x = blockIdx.y * blockDim.y * 2 + threadIdx.y;
	const int y = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	const int c_out = blockIdx.z;

	const int batch_size = error.dim(0),
		output_channels = error.dim(1),
		width = error.dim(2),
		height = error.dim(3);

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;
	const int threads_in_block = blockDim.x * blockDim.y;
	const bool val1inbound = x < width&& y < height;
	const bool val2inbound = x + blockDim.y < width&& y + blockDim.x < height;

	extern __shared__ float sm[];

	float val = 0.;
	for (int b = 0; b < batch_size; b++) {
		if (val1inbound) val += error(b, c_out, x, y);
		if (val2inbound) val += error(b, c_out, x + blockDim.y, y + blockDim.x);
	}
	sm[tid] = val;

	__syncthreads();

	int n = threads_in_block;
	while (n > 1) {
		n /= 2;
		if (tid < n)
			sm[tid] += sm[tid + n];

		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(&gradient_bias(c_out), sm[tid]);
}

Tensor<float, 4> ConvReLU::forward(const Tensor<float, 4>& input) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = input.dim(0),
		width = input.dim(2),
		height = input.dim(3);

	assert(input.dim(1) == input_channels);

	this->input = new Tensor<float, 4>(input);

	this->output = new Tensor<float, 4>(batch_size, output_channels, width, height);
	getGridSize(gridDim, blockDim, output_channels, width, height);
	switch (filter_size) {
	case 3:
		convolution_relu_forward<3> << <gridDim, blockDim >> > (*this->input, *this->output, weights, bias);
		break;
	case 5:
		convolution_relu_forward<5> << <gridDim, blockDim >> > (*this->input, *this->output, weights, bias);
		break;
	default:
		assert(false); 
	}
	return *this->output;
}

Tensor<float, 4> ConvReLU::backward(const Tensor<float, 4>& error) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = error.dim(0),
		width = error.dim(2),
		height = error.dim(3);

	printf("ConvReLU::backward: batch_size=%d, input_channels=%d, output_channels=%d, width=%d, height=%d\n",
		batch_size, input_channels, output_channels, width, height);
	printf("Input dims: %d %d %d %d\n", this->input->dim(0), this->input->dim(1), this->input->dim(2), this->input->dim(3));
	assert(error.dim(1) == output_channels);
	assert(input->dim(2) == width);
	assert(input->dim(3) == height);

	getGridSize(gridDim, blockDim, output_channels, width, height);
	relu_backward << <gridDim, blockDim >> > (error, *this->output);

	Tensor<float, 4> next_error(batch_size, input_channels, width, height);
	getGridSize(gridDim, blockDim, input_channels, width, height);
	switch (filter_size) {
	case 3:
		convolution_backward<3> << <gridDim, blockDim >> > (error, next_error, weights);
		break;
	case 5:
		convolution_backward<5> << <gridDim, blockDim >> > (error, next_error, weights);
		break;
	default:
		assert(false);
	}

	Tensor<float, 4> gradient_weights(output_channels, input_channels, filter_size, filter_size);
	Tensor<float, 1> gradient_bias(output_channels);
	gradient_weights.setZero();
	gradient_bias.setZero();

	// Grid and block sizes for parallel reduction
	blockDim.x = width < 32 ? width : 32;
	blockDim.y = height < 32 ? height : 32;
	blockDim.z = 1;
	gridDim.x = (((height + blockDim.x - 1) / blockDim.x) / 2) * filter_size;
	gridDim.y = (((width + blockDim.y - 1) / blockDim.y) / 2) * filter_size;
	gridDim.z = output_channels * input_channels;
	assert(gridDim.z < 65536);

	// Special case for 16x16 or smaller images
	if (gridDim.x == 0) gridDim.x = 1;
	if (gridDim.y == 0) gridDim.y = 1;
	size_t sm = blockDim.x * blockDim.y * sizeof(float);

	switch (filter_size) {
	case 3:
		convolution_gradient_weights_reduction<3> << <gridDim, blockDim, sm >> > (*input, error, gradient_weights);
		break;
	case 5:
		convolution_gradient_weights_reduction<5> << <gridDim, blockDim, sm >> > (*input, error, gradient_weights);
		break;
	default:
		assert(false);
	}

	gridDim.x /= filter_size;
	gridDim.y /= filter_size;
	gridDim.z = output_channels;
	convolution_gradient_bias_reduction << <gridDim, blockDim, sm >> > (error, gradient_bias);

	optimizer->update(weights, bias, gradient_weights, gradient_bias);
	delete this->input;
	delete this->output;
	this->input = nullptr;
	this->output = nullptr;
	return next_error;
}

// #define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line) {
// 	if (code != cudaSuccess) {
// 		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
// 		exit(code);
// 	}
// }

extern "C" {

	void* create_conv_relu(int in_channels, int out_channels, int filter_size) {
		return reinterpret_cast<void*>(new ConvReLU(in_channels, out_channels, filter_size));
	}

	void destroy_conv_relu(void* ptr) {
		delete reinterpret_cast<ConvReLU*>(ptr);
	}

	void conv_relu_forward(void* ptr, float* input_ptr, float* output_ptr, int N, int C, int H, int W) {
		ConvReLU* conv_relu = reinterpret_cast<ConvReLU*>(ptr);

		// Construct input tensor (device memory will be allocated internally)
		Tensor<float, 4> input({N, C, H, W});
		cudaErrchk(cudaMemcpy(input.get_dev_data(), input_ptr, sizeof(float) * input.num_elements(), cudaMemcpyHostToDevice));

		// Run forward pass
		Tensor<float, 4> output = conv_relu->forward(input);
		output.moveToHost();  // copy device to host for .data

		std::memcpy(output_ptr, output.get_data(), sizeof(float) * output.num_elements());
	}

	void conv_relu_backward(void* ptr, float* grad_output_ptr, float* grad_input_ptr, int N, int C, int H, int W) {
		ConvReLU* conv_relu = reinterpret_cast<ConvReLU*>(ptr);

		Tensor<float, 4> grad_output({N, C, H, W});
		cudaErrchk(cudaMemcpy(grad_output.get_dev_data(), grad_output_ptr, sizeof(float) * grad_output.num_elements(), cudaMemcpyHostToDevice));

		Tensor<float, 4> grad_input = conv_relu->backward(grad_output);
		grad_input.moveToHost();

		std::memcpy(grad_input_ptr, grad_input.get_data(), sizeof(float) * grad_input.num_elements());
	}

}