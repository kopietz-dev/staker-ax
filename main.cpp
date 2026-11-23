#include "include/drizzle.h"
#include "include/processer.h"
#include "include/utils.h"
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <print>
#include <vector>

void shiftImg(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, cv::Point2d shift,
              cv::cuda::Stream &stream = cv::cuda::Stream::Null()) {
  cv::Matx23d M(1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
  cv::cuda::warpAffine(src, dst, M, src.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(), stream);
}

void displayM(const cv::Mat &src) {
  std::cout << "[";
  for (int i = 0; i < src.cols * src.rows; i++) {
    if (i != 0)
      std::cout << " ,";
    std::cout << src.at<cv::Complexf>(i);
  }
  std::cout << "]\n";
}

void toComplex(const cv::Mat &src, cv::Mat &dst) {
  cv::Mat planes[] = {src, cv::Mat::zeros(src.size(), CV_32F)};
  cv::merge(planes, 2, dst);
}

cv::Mat toComplex(const cv::Mat &src) {
  cv::Mat dst;
  toComplex(src, dst);
  return dst;
}

void toReal(const cv::Mat &src, cv::Mat &dst) {
  cv::extractChannel(src, dst, 0);
}

cv::Mat toReal(const cv::Mat &src) {
  cv::Mat dst;
  cv::extractChannel(src, dst, 0);
  return dst;
}

void toHalf(const cv::Mat &full, cv::Mat &half) {
  full(cv::Rect(0, 0, full.cols / 2 + 1, full.rows)).copyTo(half);
}

int main() {
  // TODO: crop for planetary images
  // TODO: double pass for better accuracy (or maybe triple!!!)
  // TODO: make zoom_dft work with cuda output (compressed output)
  // TODO: add frequency weighting (high freq, SNR taper or alternatively add
  // filter based on known camera params (noise)
  auto batch = loadBatch("/home/kopietz/moon3");

  auto shifts = findSubpixelShifts(batch, batch[0]);

  std::vector<cv::cuda::GpuMat> shifted(batch.size());

  for (int i = 0; i < batch.size(); i++) {
    shiftImg(batch[i], shifted[i], shifts[i]);
  }

  cv::cuda::GpuMat stacked = stackSimple(shifted);
  shifts = findSubpixelShifts(batch, stacked);

  int scale = 2;

  cv::Size outSize = batch[0].size() * scale;

  cv::cuda::GpuMat accumulator(outSize, CV_32F, cv::Scalar(0));
  cv::cuda::GpuMat weighMap(outSize, CV_32F, cv::Scalar(0));

  cv::cuda::Stream streams[4];

  std::cout << "cudastart\n";
  for (int i = 0; i < batch.size(); i++) {
    drizzle(batch[i], accumulator, weighMap, 1.0f, scale, shifts[i],
            streams[i % 4]);
  }

  for (int i = 0; i < 4; i++) {
    streams[i].waitForCompletion();
  }

  std::cout << "cudastop\n";

  cv::cuda::GpuMat normalized;
  // TODO: fix zero division
  cv::cuda::divide(accumulator, weighMap, normalized);

  display(normalized);

  return 0;
}
