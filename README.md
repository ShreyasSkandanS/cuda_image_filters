# CUDA Image Filters on NVIDIA TX-2 & NVIDIA GTX1070
Implementation of GPU accelerated Median Filter and Bilateral Filter for Stereo Vision using CUDA and OpenCV for CIS601 - Special Topics in Computer Architecture : GPGPU Programming

To use this repository, make sure you have the following components installed:
* OpenCV 2.4
* CUDA 8.0

Clone the repository;

Once inside the repository launch the Makefile with the "make" command;

This repository already contains some test data in the "/data" folder; If you wish to use your own images, add a link to your desired image in *main.cpp* located in the "/src" folder;

Performance is measured over 10 attempts, and the first warm-up kernel launch is omitted in each case. You can change this by modifying the **attempts** variable in *main.cpp*;

To run this code:

> $ cd /path/to/repository/cuda-image-filters

> $ make

> $ ./build/filters_gpu

# Results

*Window Size* for both filters is **9**

*Sigma1* for Bilateral filter is **50px**

*Sigma2* for Bilateral filter is **50px**

## Median Filter

Original Image:

![Original Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/imagemedian.png)

CUDA Implementation:

![CUDA Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/gpu_median_result.png)

OpenCV Implementation:

![OpenCV Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/cpu_median_result.png)


## Bilateral Filter

Original Image:

![Original Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/imagebilateral.png)

CUDA Implementation:

![CUDA Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/gpu_bilateral_result.png)

OpenCV Implementation:

![OpenCV Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/cpu_bilateral_result.png)


# Performance

All performance times are measured in **milliseconds** (Y-Axis). Image resolution during performance analysis was (480 x 1200).

## Median Filter

Naive Implementation
![Median Filter Naive](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/image%20(2).png)

Shared Memory Implementation
![Median Filter SM](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/image%20(3).png)

## Bilateral Filter

![Bilateral Filter](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/image%20(4).png)

If you have any questions or find any mistakes in this repository please raise an issue or contact me.

