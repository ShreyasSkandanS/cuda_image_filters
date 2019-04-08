#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "helper_cuda.h"

const int BLOCKDIM = 32;
const int sigma1 = 50;
const int sigma2 = 50;

__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

__device__ int index_median(int x, int y, int width) 
{
	return (y * width) + x;
}

__device__ int clamp_median(int value, int bound) 
{
	if (value < 0) {
		return 0;
	}
	if (value < bound) {
		return value;
	}
	return bound - 1;
}

__device__ float exp(int i) { return exp((float) i); }


__global__ void bilateral_filter_sm(unsigned char* input, unsigned char* output, int width, int height)
{
	__shared__ float sharedPixels[BLOCKDIM + FILTER_SIZE][BLOCKDIM + FILTER_SIZE];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int xBlockLimit_max = blockDim.x - FILTER_HALFSIZE - 1;
	int yBlockLimit_max = blockDim.y - FILTER_HALFSIZE - 1;
	int xBlockLimit_min = FILTER_HALFSIZE;
	int yBlockLimit_min = FILTER_HALFSIZE;

	if (threadIdx.x > xBlockLimit_max && threadIdx.y > yBlockLimit_max) {
		int i = index_median(clamp_median(x + FILTER_HALFSIZE,width), clamp_median(y + FILTER_HALFSIZE,height), width);
	    	float pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x > xBlockLimit_max && threadIdx.y < yBlockLimit_min) {
		int i = index_median(clamp_median(x + FILTER_HALFSIZE,width), clamp_median(y - FILTER_HALFSIZE,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min && threadIdx.y > yBlockLimit_max) {
		int i = index_median(clamp_median(x - FILTER_HALFSIZE,width), clamp_median(y + FILTER_HALFSIZE,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min && threadIdx.y < yBlockLimit_min) {
		int i = index_median(clamp_median(x - FILTER_HALFSIZE,width), clamp_median(y - FILTER_HALFSIZE,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min) {
		int i = index_median(clamp_median(x - FILTER_HALFSIZE,width), clamp_median(y,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y + FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x > xBlockLimit_max) {
		int i = index_median(clamp_median(x + FILTER_HALFSIZE,width), clamp_median(y,height), width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.y < yBlockLimit_min) {
		int i = index_median(clamp_median(x,width), clamp_median(y - FILTER_HALFSIZE,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y] = pixel;
	}
	if (threadIdx.y > yBlockLimit_max) {
		int i = index_median(clamp_median(x,width), clamp_median(y + FILTER_HALFSIZE,height), width);
		float pixel = input[i];
		sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	int i = index_median(x, y, width);
	float pixel = input[i];
	sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;

	__syncthreads();

	if((x<width) && (y<height))
	{
		float running_total = 0;
		float norm_factor = 0;
		const int offset = y * width + x;
		for (int xctr = 0; xctr < FILTER_SIZE; xctr++) 
		{
			for (int yctr = 0; yctr < FILTER_SIZE; yctr++) 
			{
				float intensity_change = sharedPixels[threadIdx.x + xctr][threadIdx.y + yctr] - sharedPixels[threadIdx.x][threadIdx.y];
				float w1 = exp(-((xctr-FILTER_HALFSIZE) * (xctr-FILTER_HALFSIZE) + 
						 (yctr-FILTER_HALFSIZE) * (yctr-FILTER_HALFSIZE)) / 
						(2 * sigma1 * sigma1));
				float w2 = exp(-(intensity_change * intensity_change) / (2 * sigma2 * sigma2));
				running_total += sharedPixels[threadIdx.x + xctr][threadIdx.y + yctr] * w1 * w2;
				norm_factor += w1 * w2;
			}
		}
		output[offset] = running_total / norm_factor;
	}
}


__global__ void bilateral_filter_2d(unsigned char* input, unsigned char* output, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x<width) && (y<height))
	{
		float running_total = 0;
		float norm_factor = 0;
		const int offset = y * width + x;
		for (int xctr = -FILTER_HALFSIZE; xctr <= FILTER_HALFSIZE; xctr++) 
		{
			for (int yctr = -FILTER_HALFSIZE; yctr <= FILTER_HALFSIZE; yctr++) 
			{
				int y_iter = y + xctr;
				int x_iter = x + yctr;
				if (x_iter < 0) x_iter = -x_iter;
				if (y_iter < 0) y_iter = -y_iter;
				if (x_iter > width-1) x_iter = width-1-xctr;
				if (y_iter > height-1) y_iter = height-1-yctr;
				float intensity_change = input[y_iter * width + x_iter] - input[y * width + x];
				float w1 = exp(-(xctr * xctr + yctr * yctr) / (2 * sigma1 * sigma1));
				float w2 = exp(-(intensity_change * intensity_change) / (2 * sigma2 * sigma2));
				running_total += input[y_iter * width + x_iter] * w1 * w2;
				norm_factor += w1 * w2;
			}
		}
		output[offset] = running_total / norm_factor;
	}
}


__global__ void bilateral_filter_2d_unoptimized(unsigned char* input, unsigned char* output, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x<width) && (y<height))
	{
		float running_total = 0;
		float norm_factor = 0;
		const int offset = y * width + x;
		for (int xctr = -FILTER_HALFSIZE; xctr <= FILTER_HALFSIZE; xctr++) 
		{
			for (int yctr = -FILTER_HALFSIZE; yctr <= FILTER_HALFSIZE; yctr++) 
			{
				int y_iter = y + xctr;
				int x_iter = x + yctr;
				if (0 <= x_iter && x_iter < width && 0 <= y_iter && y_iter < height) 
				{
					float intensity_change = input[y_iter * width + x_iter] - input[y * width + x];
					float v1 = exp(-(xctr * xctr + yctr * yctr) / (2 * sigma1 * sigma1));
					float v2 = exp(-(intensity_change * intensity_change) / (2 * sigma2 * sigma2));
					running_total += input[y_iter * width + x_iter] * v1 * v2;
					norm_factor += v1 * v2;
				}
			}
		}
		output[offset] = running_total / norm_factor;
	}
}


void bilateral_filter_wrapper(const cv::Mat& input, cv::Mat& output)
{
	unsigned char *d_input, *d_output;
	cudaError_t cudaStatus;	
	
	cudaStatus = cudaMalloc<unsigned char>(&d_input,input.rows*input.cols);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaMalloc<unsigned char>(&d_output,output.rows*output.cols);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(d_input,input.ptr(),input.rows*input.cols,cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);	
	
	const dim3 block(BLOCKDIM,BLOCKDIM);
	const dim3 grid(input.cols/BLOCKDIM, input.rows/BLOCKDIM);

	bilateral_filter_2d_unoptimized<<<grid,block>>>(d_input,d_output,input.cols,input.rows);

	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaMemcpy(output.ptr(),d_output,output.rows*output.cols,cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);	
}
