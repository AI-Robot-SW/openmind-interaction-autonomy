//
// distmap.cu
//
// BEV 점유 격자 → 장애물까지의 거리맵(m) 변환 커널
//
// 알고리즘: BFS wave propagation (relaxation)
//   - 장애물 셀(obstacle=1)에서 거리 0으로 시작
//   - 8방향 이웃으로 거리를 전파 (대각선은 √2 배)
//   - 수렴 체크 없이 max_iter(= ceil(max_dist/res))회 고정 반복
//   - CPU-GPU 동기화 없이 커널만 반복 → 마지막 D2H에서만 동기화
//
// 입력 레이아웃: (H × W) float32  obstacle  (1.0=장애물, 0.0=자유)
// 출력 레이아웃: (H × W) float32  dist      (미터 단위, [0, max_dist])
//

extern "C"
__global__ void distmap_bfs(
    float       *dist,      // (H × W) float32, in/out
    const float *obstacle,  // (H × W) float32, read-only
    int          width,
    int          height,
    float        res,
    float        max_dist
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // col — X(좌우) 방향
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row — Z(전방) 방향
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 장애물 셀은 거리 0 고정, 전파 대상 아님
    if (obstacle[idx] > 0.5f) return;

    float min_dist = dist[idx];

    // 8방향 이웃에서 전파된 거리의 최솟값을 구함
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            // 대각선 이동은 √2 × res, 직선은 1 × res
            float step = res * ((dx != 0 && dy != 0) ? 1.4142135f : 1.0f);
            float cand = dist[ny * width + nx] + step;
            if (cand < min_dist) min_dist = cand;
        }
    }

    dist[idx] = fminf(min_dist, max_dist);
}
