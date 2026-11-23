#include "drizzle.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

__global__ void drizzleKernel(cv::cuda::PtrStepSz<float> src,
                              cv::cuda::PtrStepSz<float> accumulator,
                              cv::cuda::PtrStepSz<float> weightMap,
                              float shiftx, float shifty, float halfPixsize,
                              float scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float xf = static_cast<float>(x);
  const float yf = static_cast<float>(y);

  if (x >= src.cols || y >= src.rows)
    return;

  const float originx = (xf + 0.5f + shiftx) * scale;
  const float originy = (yf + 0.5f + shifty) * scale;

  const int p0rx = __float2int_rd(originx - halfPixsize);
  const int p0ry = __float2int_rd(originy - halfPixsize);
  const int p1rx = __float2int_ru(originx + halfPixsize);
  const int p1ry = __float2int_ru(originy + halfPixsize);

  const float value = src(y, x);

  for (int i = p0rx; i < p1rx; i++) {
    const float p0lx = max(static_cast<float>(i), originx - halfPixsize);
    const float p1lx = min(static_cast<float>(i), originx + halfPixsize);
    const float localWidth = (p1lx - p0lx);

    for (int j = p0ry; j < p1ry; j++) {
      if (i < 0 || i >= accumulator.cols || j < 0 ||
          j >= accumulator.rows) // TODO: this could be optimised prob,
                                 // change xy outer loop
        continue;

      const float p0ly = max(static_cast<float>(j), originy - halfPixsize);
      const float p1ly = min(static_cast<float>(j), originy + halfPixsize);

      const float localHeight = (p1ly - p0ly);

      const float weight =
          localHeight * localWidth / (halfPixsize * halfPixsize);
      // should be also division by 4, but during the
      // normalistation it does not matter
      // tbh, the division by halPixSize^2 does not matter too, but it
      // normalises the weight to [0, 1]

      atomicAdd(&weightMap(j, i), weight);
      atomicAdd(&accumulator(j, i), weight * value);
    }
  }
}

void drizzle(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &accumulator,
             cv::cuda::GpuMat &weightMap, float pixsize, float scale,
             cv::Point2f shift, cv::cuda::Stream &stream) {
  CV_Assert(src.type() == CV_32FC1);

  dim3 block(32, 8);
  dim3 grid((src.cols + block.x - 1) / block.x,
            (src.rows + block.y - 1) / block.y);

  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
  drizzleKernel<<<grid, block, 0, s>>>(src, accumulator, weightMap, shift.x,
                                       shift.y, pixsize * 0.5f, scale);
}
