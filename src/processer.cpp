
#include "../include/processer.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/exception.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <print>
#include <vector>

std::vector<std::pair<double, size_t>>
evaluateSharpness(const std::vector<cv::cuda::GpuMat> &batch) {
  CV_Assert(batch[0].type() == CV_32F);

  auto sobelX = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
  auto sobelY = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);

  std::array<cv::cuda::Stream, streamCount> streams;

  std::vector<cv::cuda::GpuMat> gradientX(batch.size()),
      gradientY(batch.size()), gradientMag(batch.size()),
      meanStdDev(batch.size());

  for (int i = 0; i < batch.size(); i++) {
    cv::cuda::Stream &stream = streams[i % streamCount];

    sobelX->apply(batch[i], gradientX[i], stream);
    sobelY->apply(batch[i], gradientY[i], stream);

    cv::cuda::pow(gradientX[i], 2, gradientX[i], stream);
    cv::cuda::pow(gradientY[i], 2, gradientY[i], stream);

    cv::cuda::add(gradientX[i], gradientY[i], gradientMag[i], cv::noArray(),
                  CV_32F, stream);

    cv::cuda::meanStdDev(gradientMag[i], meanStdDev[i], cv::noArray(), stream);
  }

  std::vector<std::pair<double, size_t>> scores(batch.size());

  for (int i = 0; i < batch.size(); i++) {
    cv::Mat cpuMeanStdDev;
    meanStdDev[i].download(cpuMeanStdDev);
    scores[i] = {cpuMeanStdDev.at<double>(0, 0), i};
  }

  std::sort(scores.begin(), scores.end());

  return scores;
}

cv::cuda::GpuMat createHanningWindow(cv::Size size) {
  cv::Mat hanningWindow;
  cv::createHanningWindow(hanningWindow, size, CV_32F);

  cv::cuda::GpuMat gpuHanningWindow;
  gpuHanningWindow.upload(hanningWindow);
  return gpuHanningWindow;
}

cv::cuda::GpuMat createGaussianFilter(float sigmaL, float sigmaH,
                                      cv::Size size) {
  cv::Mat gaussian(size, CV_32F);

  float w = size.width, h = size.height;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      gaussian.at<float>(y, x) =
          std::exp(-0.5f * std::powf((x - w / 2) / (w / sigmaL), 2.0f) -
                   0.5f * std::powf((y - h / 2) / (h / sigmaL), 2.0f)) *
          (1.0f - std::exp(-0.5f * std::powf((x - w / 2) / (w / sigmaH), 2.0f) -
                           0.5f * std::powf((y - h / 2) / (h / sigmaH), 2.0f)));
    }
  }

  cv::cuda::GpuMat gpuGaussian;
  gpuGaussian.upload(gaussian);

  return gpuGaussian;
}

cv::Point2d bound(cv::Point2d point, cv::Size size) {
  return cv::Point2d(
      std::min(std::max(point.x, 0.0), static_cast<double>(size.width)),
      std::min(std::max(point.y, 0.0), static_cast<double>(size.height)));
}

cv::Size getDFTSize(cv::Size size) {
  return cv::Size(size.width / 2 + 1, size.height);
}

std::vector<cv::Point2d>
findSubpixelShifts(const std::vector<cv::cuda::GpuMat> &batch,
                   const cv::cuda::GpuMat &reference) {
  cv::cuda::GpuMat hanningWindow = createHanningWindow(batch[0].size());

  const size_t n = batch.size();

  auto dft = cv::cuda::createDFT(batch[0].size(), 0);
  auto invDFT = cv::cuda::createDFT(batch[0].size(),
                                    cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT |
                                        cv::DFT_SCALE | cv::DFT_COMPLEX_INPUT);

  cv::cuda::GpuMat referenceWindowed;
  cv::cuda::multiply(reference, hanningWindow, referenceWindowed, 1, CV_32F);
  cv::cuda::GpuMat referenceFT;
  dft->compute(referenceWindowed, referenceFT);

  cv::cuda::GpuMat minMaxVals;
  std::vector<std::array<cv::cuda::GpuMat, 2>> planes(n);

  std::array<cv::cuda::Stream, streamCount> streams;
  std::vector<cv::cuda::GpuMat> buffer(n), ft(n), locs(n), buffer2(n);

  // TODO: tune this
  cv::cuda::GpuMat gaussianFilter =
      createGaussianFilter(5, 7, getDFTSize(batch[0].size()));

  cv::cuda::GpuMat gaussianPlaneBuffer[] = {gaussianFilter, gaussianFilter};
  cv::cuda::GpuMat complexGaussianFilter;
  cv::cuda::merge(gaussianPlaneBuffer, 2, complexGaussianFilter);

  for (int i = 0; i < n; i++) {
    cv::cuda::Stream &stream = streams[i % streamCount];

    cv::cuda::multiply(batch[i], hanningWindow, buffer[i], 1, CV_32F, stream);

    dft->compute(buffer[i], ft[i], stream);

    cv::cuda::mulSpectrums(ft[i], referenceFT, buffer[i], 0, true, stream);

    // apply hi and lo pass filter
    // std::println("type {} ?= {}", ft[i].type(),
    // complexGaussianFilter.type()); std::println("size {} ?= {}",
    // ft[i].size().width, complexGaussianFilter.size().width);
    cv::cuda::multiply(ft[i], complexGaussianFilter, ft[i], 1, CV_32F, stream);

    cv::cuda::split(buffer[i], planes[i].data(), stream);

    cv::cuda::magnitude(planes[i][0], planes[i][1], buffer[i], stream);

    cv::cuda::add(buffer[i], 1e-6, buffer[i], cv::noArray(), -1, stream);

    cv::cuda::divide(planes[i][0], buffer[i], planes[i][0], 1.0, -1, stream);
    cv::cuda::divide(planes[i][1], buffer[i], planes[i][1], 1.0, -1, stream);

    cv::cuda::merge(planes[i].data(), 2, buffer2[i], stream);

    invDFT->compute(buffer2[i], buffer[i], stream);

    cv::cuda::findMinMaxLoc(buffer[i], minMaxVals, locs[i], cv::noArray(),
                            stream);
  }

  for (cv::cuda::Stream &stream : streams)
    stream.waitForCompletion();

  int imgWidth = batch[0].cols;
  int imgHeight = batch[0].rows;

  std::vector<cv::Point2d> maxLocations;

  // get coarse peak location on cpu
  for (int i = 0; i < n; i++) {
    cv::Mat locCPU;
    locs[i].download(locCPU);
    int maxIndex = locCPU.at<int>(1, 0);

    maxLocations.emplace_back(maxIndex % imgWidth, maxIndex / imgWidth);
  }

  std::vector<cv::Size> upscaledSize;

  for (int i = 0; i < n; i++) {
    // this takes millenia to run (bottleneck)
    cv::Mat correlationSurface;
    buffer2[i].download(correlationSurface);

    // TODO: this must be on GPU (too slow)
    cv::Point2f max = findMaximum(correlationSurface, maxLocations[i]);

    maxLocations[i] = max;

    if (maxLocations[i].x >= batch[i].cols / 2)
      maxLocations[i].x -= batch[i].cols;

    if (maxLocations[i].y >= batch[i].rows / 2)
      maxLocations[i].y -= batch[i].rows;

    maxLocations[i] *= -1;
  }

  return maxLocations;
}

cv::cuda::GpuMat stackSimple(const std::vector<cv::cuda::GpuMat> &batch) {
  cv::cuda::GpuMat accumulator = batch[0].clone();

  for (int i = 1; i < batch.size(); i++) {
    cv::cuda::add(accumulator, batch[i], accumulator);
  }

  cv::cuda::divideWithScalar(accumulator, static_cast<float>(batch.size()),
                             accumulator);

  return accumulator;
}

std::vector<cv::cuda::GpuMat> loadBatch(const std::string &directory) {
  std::vector<cv::cuda::GpuMat> batch;

  for (auto &file : std::filesystem::directory_iterator(directory)) {
    if (batch.size() > 1000)
      break;

    if (!file.is_directory() && file.path().extension().string() == ".tiff") {

      cv::Mat img = cv::imread(file.path().string(), cv::IMREAD_GRAYSCALE);

      if (img.type() != CV_32F) {
        img.convertTo(img, CV_32F, 1.0f / 255.0f);
      }

      cv::cuda::GpuMat gpuImg;
      gpuImg.upload(img);

      batch.emplace_back(gpuImg);
    }
  }

  return batch;
}
