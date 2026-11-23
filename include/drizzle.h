#pragma once
#include "../include/utils.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

void drizzle(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &accumulator,
             cv::cuda::GpuMat &weightMap, float pixsize, float scale,
             cv::Point2f shift,
             cv::cuda::Stream &stream = cv::cuda::Stream::Null());
