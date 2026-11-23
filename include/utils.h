#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

inline cv::Complexf polar(float phase) {
  return cv::Complexf(std::cos(phase), std::sin(phase));
}

inline cv::Point2f max(cv::Point2f a, cv::Point2f b) {
  return cv::Point2f(std::max(a.x, b.x), std::max(a.y, b.y));
}

inline cv::Point2f floor(cv::Point2f a) {
  return cv::Point2f(std::floor(a.x), std::floor(a.y));
}

inline cv::Point2f ceil(cv::Point2f a) {
  return cv::Point2f(std::ceil(a.x), std::ceil(a.y));
}

inline cv::Point2f min(cv::Point2f a, cv::Point2f b) {
  return cv::Point2f(std::min(a.x, b.x), std::min(a.y, b.y));
}

inline cv::Point2f newtonStep(const cv::Mat &src, cv::Point2f s0) {
  const float M = static_cast<float>(src.cols);
  const float W = (M - 1) * 2.0f;
  const float N = static_cast<float>(src.rows);

  // calc gradient
  cv::Complexf k_gradient;
  cv::Complexf l_gradient;

  cv::Complexf hessiankk;
  cv::Complexf hessianll;
  cv::Complexf hessiankl;

  // TODO: problably can be parralelised
  //  can be expressed with matrix multiplication
  //  can be brought to gpu

  for (int k = 0; k < N; k++) {
    for (int l = 0; l < M; l++) {
      int kk = (k < N / 2) ? k : k - N;
      int ll = l;

      float omega_k = 2.0f * CV_PI * static_cast<float>(kk) / N;
      float omega_l = 2.0f * CV_PI * static_cast<float>(ll) / W;

      k_gradient += omega_k * src.at<cv::Complexf>(k, l) *
                    cv::Complexf(0, 1.0f) *
                    polar(omega_l * s0.x + omega_k * s0.y);

      l_gradient += omega_l * src.at<cv::Complexf>(k, l) *
                    cv::Complexf(0, 1.0f) *
                    polar(omega_l * s0.x + omega_k * s0.y);

      hessiankk += omega_k * omega_k * src.at<cv::Complexf>(k, l) *
                   polar(omega_l * s0.x + omega_k * s0.y);

      hessianll += omega_l * omega_l * src.at<cv::Complexf>(k, l) *
                   polar(omega_l * s0.x + omega_k * s0.y);

      hessiankl += omega_k * omega_l * src.at<cv::Complexf>(k, l) *
                   polar(omega_l * s0.x + omega_k * s0.y);
    }
  }

  cv::Mat hessian(2, 2, CV_32F);
  cv::Mat gradient(2, 1, CV_32F);

  gradient.at<float>(0, 0) = l_gradient.re;
  gradient.at<float>(1, 0) = k_gradient.re;

  hessian.at<float>(0, 0) = hessianll.re;
  hessian.at<float>(0, 1) = hessiankl.re;
  hessian.at<float>(1, 1) = hessiankk.re;
  hessian.at<float>(1, 0) = hessiankl.re;

  hessian *= -1.0f;

  cv::Mat t(2, 1, CV_32F);

  t.at<float>(0, 0) = s0.x;
  t.at<float>(1, 0) = s0.y;

  // TODO
  // hessian is 2x2 add closed form inverse
  //  guard for converge (10e-4, guard for hessian, limit steps to 0.25px)
  // reject when the finish is too far from the x0
  cv::Mat tt = t - hessian.inv() * gradient;

  return cv::Point2f(tt.at<float>(0, 0), tt.at<float>(1, 0));
}

inline cv::Point2f findMaximum(const cv::Mat &src, cv::Point2f x0,
                               int maxIter = 10) {
  cv::Point2f currentEstimate = x0;

  for (int i = 0; i < maxIter; i++) {
    currentEstimate = newtonStep(src, currentEstimate);
  }

  return currentEstimate;
}

inline void display(const cv::Mat &img) {
  cv::imshow("img", img);
  while (cv::waitKey() != 27)
    ;
}

inline void display(const cv::cuda::GpuMat &img) {
  cv::Mat cpuImg;
  img.download(cpuImg);

  display(cpuImg);
}

// only tiff support
inline void save(const std::string &filename, const cv::cuda::GpuMat &img) {
  cv::Mat cpuImg;
  img.download(cpuImg);

  cv::imwrite(
      filename, cpuImg,
      {cv::IMWRITE_TIFF_COMPRESSION, cv::IMWRITE_TIFF_COMPRESSION_NONE});
}
