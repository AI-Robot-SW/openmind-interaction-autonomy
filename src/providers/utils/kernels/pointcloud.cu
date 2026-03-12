//
// pointcloud.cu
//
// depth + RGB → XYZ RGB 포인트 클라우드 변환 커널
//
// 출력 레이아웃: 픽셀당 float4 (X, Y, Z, RGB-packed-as-float)
// 유효하지 않은 픽셀 (d <= 0 또는 d > range_max): X/Y/Z = NaN
//

#include <math.h>

extern "C"
__global__ void depth_rgb_to_xyzrgb_kernel(
    const float*         depth,
    const unsigned char* rgb,
    float*               out,
    int                  width,
    int                  height,
    float                fx,
    float                fy,
    float                cx,
    float                cy,
    int                  rgb_step,
    int                  red_offset,
    int                  green_offset,
    int                  blue_offset,
    float                range_max
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float d = depth[idx];
    float X, Y, Z;

    if (d <= 0.0f || (range_max > 0.0f && d > range_max)) {
        X = NAN;
        Y = NAN;
        Z = NAN;
    } else {
        int u = idx % width;
        int v = idx / width;
        Z = d;
        X = ((float)u - cx) * d / fx;
        Y = ((float)v - cy) * d / fy;
    }

    int rgb_idx = idx * rgb_step;
    unsigned char r = rgb[rgb_idx + red_offset];
    unsigned char g = rgb[rgb_idx + green_offset];
    unsigned char b = rgb[rgb_idx + blue_offset];

    unsigned int rgb_packed = ((unsigned int)r << 16)
                            | ((unsigned int)g <<  8)
                            | ((unsigned int)b);

    int base = idx * 4;
    out[base + 0] = X;
    out[base + 1] = Y;
    out[base + 2] = Z;
    out[base + 3] = __int_as_float(rgb_packed);
}
