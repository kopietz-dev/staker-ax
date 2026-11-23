#pragma once

#include <filesystem>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

constexpr size_t streamCount = 4;

std::vector<std::pair<double, size_t>>
evaluateSharpness(const std::vector<cv::cuda::GpuMat> &batch);

cv::cuda::GpuMat createHanningWindow(cv::Size size);
cv::cuda::GpuMat createGaussianFilter(float sigmaL, float sigmaH,
                                      cv::Size size);

std::vector<cv::Point2d>
findSubpixelShifts(const std::vector<cv::cuda::GpuMat> &batch,
                   const cv::cuda::GpuMat &reference);

std::vector<cv::cuda::GpuMat> loadBatch(const std::string &directory);

cv::cuda::GpuMat stackSimple(const std::vector<cv::cuda::GpuMat> &batch);
