//
// bev.cu
//
// XYZ-RGB 포인트 클라우드 (N×4 float32) → Bird's-Eye-View BGR 이미지 변환 커널
//
// 입력 레이아웃: 포인트당 float4 (X, Y, Z, RGB-packed-as-float)
// 출력 레이아웃: (height × width × 3) uint8 BGR
//

extern "C"
__global__ void bev_kernel(
    const float   *points,     // (N, 4): [X, Y, Z, RGB-packed-as-float]
    int            num_points,
    int            width,
    int            height,
    float          res,
    unsigned char *bev_img     // (height * width * 3) BGR
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = points[idx * 4 + 0];
    float z = points[idx * 4 + 2];

    int gx = (int)(x / res + width  / 2);
    int gz = (int)(height - z / res);

    if (gx < 0 || gx >= width || gz < 0 || gz >= height) return;

    unsigned int rgb = __float_as_int(points[idx * 4 + 3]);
    unsigned char r = (rgb >> 16) & 0xFF;
    unsigned char g = (rgb >>  8) & 0xFF;
    unsigned char b =  rgb        & 0xFF;

    int offset = (gz * width + gx) * 3;
    bev_img[offset + 0] = b;
    bev_img[offset + 1] = g;
    bev_img[offset + 2] = r;
}
