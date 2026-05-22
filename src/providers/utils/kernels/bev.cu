//
// bev_pipeline.cu
//
// BEV occupancy grid 전체 파이프라인을 GPU에서 실행한다.
//
// 커널 실행 순서 (Python 쪽에서 순서대로 launch):
//   1. pc_to_grids_kernel          pointcloud(N,4) → free_ep / avoid / curb / person 그리드
//   2. make_blocked_kernel         avoid|curb|person → blocked 그리드
//   3. blocker_bottom_kernel       semantic_map → 열별 장애물 하단 행 (with dilation)
//   4. ground_proj_kernel          semantic_map + blocker_bottom → ground_free 그리드
//   5. dilate_3x3_kernel           (ground_free → ground_dil)
//   6. max_grids_kernel            max(free_ep, ground_dil) → free_cand
//   7. raycast_grid_kernel         Bresenham ray cast (한 thread = 한 grid cell)
//   8. dilate_3x3_kernel           (ray_grid → temp)
//   9. dilate_5x5_kernel           (temp → ray_grid)     ─┐ morphological
//  10. erode_5x5_kernel            (ray_grid → temp)     ─┘ close(5x5)
//  11. merge_final_kernel          (temp + obstacle grids) → grid_np int8
//  12. ray_gap_fill_final_kernel   (grid_np in-place) 카메라 시점 기반 노이즈 채우기
//        sensor→경계 방향 광선 위에서 free→avoid/curb(≤N셀)→free(≥M셀) 패턴 감지 시 채움
//        person(88) 뒤로는 채우지 않음
//
// 모든 커널은 동일 스트림(기본 스트림)에서 순서대로 실행되므로
// 커널 간 명시적 동기화는 불필요하다.
//

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// 1. Pointcloud → {free_ep, avoid, curb, person} 그리드
//
//   points : (N, 4) float32  AoS — [x_right, y, z_fwd, rgb_packed]
//   semantic color 기준 (packed RGB uint32):
//     driveable : g>100, r<80,  b<80
//     person    : b>100, r<80,  g<80
//     avoid     : r>200, g<80,  b<80
//     curb      : r>200, g>200, b>200
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pc_to_grids_kernel(
    const float         *__restrict__ points,
    int                  N,
    float                inv_res,
    float                dx_minus_ox,   // dx - origin_x
    float                dy_minus_oy,   // dy - origin_y
    int                  W,
    int                  H,
    unsigned char       *free_ep_grid,
    unsigned char       *avoid_grid,
    unsigned char       *curb_grid,
    unsigned char       *person_grid
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float        *p      = points + idx * 4;
    float               x_right = p[0];
    float               z_fwd   = p[2];
    unsigned int        rgb     = __float_as_uint(p[3]);

    int j = (int)((z_fwd   + dx_minus_ox) * inv_res);
    int i = (int)((-x_right + dy_minus_oy) * inv_res);
    if (i < 0 || i >= H || j < 0 || j >= W) return;

    unsigned int r = (rgb >> 16) & 0xFFu;
    unsigned int g = (rgb >>  8) & 0xFFu;
    unsigned int b =  rgb        & 0xFFu;

    int cell = i * W + j;

    if      (g > 100u && r <  80u && b <  80u) free_ep_grid[cell] = 1;
    else if (b > 100u && r <  80u && g <  80u) person_grid [cell] = 1;
    else if (r > 200u && g <  80u && b <  80u) avoid_grid  [cell] = 1;
    else if (r > 200u && g > 200u && b > 200u) curb_grid   [cell] = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. blocked = avoid | curb | person
// ─────────────────────────────────────────────────────────────────────────────
__global__ void make_blocked_kernel(
    const unsigned char *__restrict__ avoid_grid,
    const unsigned char *__restrict__ curb_grid,
    const unsigned char *__restrict__ person_grid,
    unsigned char       *blocked_grid,
    int                  N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    blocked_grid[idx] = (avoid_grid[idx] | curb_grid[idx] | person_grid[idx]) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. 열별 장애물 하단 행 계산 (5×5 dilation 효과 포함)
//
//   각 장애물 픽셀 (i, j)에 대해
//   열 j-r … j+r 에 atomicMax(i + r) 기록.
//   → 5×5 dilation 후 column-wise argmax와 동등.
//
//   blocker_bottom_by_col : (W_cam,) int32, 호출 전 -1 로 초기화.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void blocker_bottom_kernel(
    const unsigned char *__restrict__ semantic_map,   // (H_cam, W_cam) uint8
    int                  H_cam,
    int                  W_cam,
    int                  dilation_radius,             // = 2 for 5×5 kernel
    int                 *blocker_bottom_by_col        // (W_cam,) int32
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= W_cam || i >= H_cam) return;

    unsigned char cls = semantic_map[i * W_cam + j];
    if (cls != 2 && cls != 3 && cls != 4) return;   // person=2, avoid=3, curb=4

    int extended_row = min(i + dilation_radius, H_cam - 1);
    for (int dj = -dilation_radius; dj <= dilation_radius; dj++) {
        int nj = j + dj;
        if (nj >= 0 && nj < W_cam)
            atomicMax(&blocker_bottom_by_col[nj], extended_row);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Driveable 픽셀을 바닥 평면에 투영 → ground_free 그리드
//
//   camera optical frame : X=right, Y=down, Z=forward
//   robot frame          : x=forward, y=left, z=up
//   변환 (카메라 수평 가정):
//     ray_x_fwd  = 1              (cam Z)
//     ray_y_left = -cam_ray_x     (−cam X)
//     ray_z_up   = -cam_ray_y     (−cam Y)
//   바닥(z=0) 교점: scale = camera_height / (−ray_z_up)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void ground_proj_kernel(
    const unsigned char *__restrict__ semantic_map,
    const int           *__restrict__ blocker_bottom_by_col,
    int                  H_cam,
    int                  W_cam,
    float                fx,
    float                fy,
    float                cx,
    float                cy,
    float                camera_height_m,
    float                dx,
    float                dy,
    float                origin_x,
    float                origin_y,
    float                inv_res,
    int                  W_grid,
    int                  H_grid,
    int                  proj_stride,
    int                  occlusion_margin,
    unsigned char       *ground_free_grid
)
{
    // 각 thread = proj_stride 로 서브샘플된 픽셀 하나
    int sj = blockIdx.x * blockDim.x + threadIdx.x;
    int si = blockIdx.y * blockDim.y + threadIdx.y;

    int W_samp = (W_cam + proj_stride - 1) / proj_stride;
    int H_samp = (H_cam + proj_stride - 1) / proj_stride;
    if (sj >= W_samp || si >= H_samp) return;

    int u_int = sj * proj_stride;
    int v_int = si * proj_stride;
    if (u_int >= W_cam || v_int >= H_cam) return;
    if (semantic_map[v_int * W_cam + u_int] != 1) return;   // driveable=1 만

    // 폐색 필터: 같은 열에 장애물이 있고 현재 픽셀이 그 위쪽이면 제외
    int bot = blocker_bottom_by_col[u_int];
    if (bot >= 0 && v_int <= bot + occlusion_margin) return;

    float cam_ray_x = ((float)u_int - cx) / fx;
    float cam_ray_y = ((float)v_int - cy) / fy;
    float ray_z_up  = -cam_ray_y;
    if (ray_z_up >= -1e-4f) return;   // 지평선 이상 → 바닥 교점 없음

    float scale         = camera_height_m / (-ray_z_up);
    float ground_x_fwd  = dx + scale;                  // ray_x_fwd = 1
    float ground_y_left = dy + scale * (-cam_ray_x);

    int j_grid = (int)((ground_x_fwd  - origin_x) * inv_res);
    int i_grid = (int)((ground_y_left - origin_y) * inv_res);
    if (i_grid < 0 || i_grid >= H_grid || j_grid < 0 || j_grid >= W_grid) return;

    ground_free_grid[i_grid * W_grid + j_grid] = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. 3×3 이진 팽창 (binary dilation)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void dilate_3x3_kernel(
    const unsigned char *__restrict__ in,
    unsigned char       *out,
    int                  W,
    int                  H
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= W || i >= H) return;

    unsigned char val = 0;
    #pragma unroll
    for (int di = -1; di <= 1; di++) {
        #pragma unroll
        for (int dj = -1; dj <= 1; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < H && nj >= 0 && nj < W)
                val |= in[ni * W + nj];
        }
    }
    out[i * W + j] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. element-wise max of two grids
// ─────────────────────────────────────────────────────────────────────────────
__global__ void max_grids_kernel(
    const unsigned char *__restrict__ a,
    const unsigned char *__restrict__ b,
    unsigned char       *out,
    int                  N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Bresenham ray cast — 한 thread = 한 grid cell (endpoint 여부 내부 판단)
//
//   free_cand 가 1인 cell이 endpoint 역할을 한다.
//   센서에서 endpoint 까지 Bresenham 선을 그리되,
//   중간에 blocked cell 을 만나면 그 뒤는 free 로 만들지 않는다.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void raycast_grid_kernel(
    const unsigned char *__restrict__ free_cand,
    const unsigned char *__restrict__ blocked,
    unsigned char       *ray_grid,
    int                  sensor_i,
    int                  sensor_j,
    int                  W,
    int                  H
)
{
    int end_j = blockIdx.x * blockDim.x + threadIdx.x;
    int end_i = blockIdx.y * blockDim.y + threadIdx.y;
    if (end_j >= W || end_i >= H) return;
    if (!free_cand[end_i * W + end_j]) return;   // endpoint 아니면 즉시 반환

    int di     = abs(end_i - sensor_i);
    int dj     = abs(end_j - sensor_j);
    int step_i = (sensor_i < end_i) ?  1 : -1;
    int step_j = (sensor_j < end_j) ?  1 : -1;
    int ci = sensor_i, cj = sensor_j;

    if (dj > di) {
        int error = dj / 2;
        while (cj != end_j) {
            if (ci < 0 || ci >= H || cj < 0 || cj >= W) return;
            if (blocked[ci * W + cj]) return;
            ray_grid[ci * W + cj] = 1;
            cj += step_j;
            error -= di;
            if (error < 0) { ci += step_i; error += dj; }
        }
    } else {
        int error = di / 2;
        while (ci != end_i) {
            if (ci < 0 || ci >= H || cj < 0 || cj >= W) return;
            if (blocked[ci * W + cj]) return;
            ray_grid[ci * W + cj] = 1;
            ci += step_i;
            error -= dj;
            if (error < 0) { cj += step_j; error += di; }
        }
    }
    // endpoint cell
    if (end_i >= 0 && end_i < H && end_j >= 0 && end_j < W
            && !blocked[end_i * W + end_j])
        ray_grid[end_i * W + end_j] = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. 5×5 이진 팽창
// ─────────────────────────────────────────────────────────────────────────────
__global__ void dilate_5x5_kernel(
    const unsigned char *__restrict__ in,
    unsigned char       *out,
    int                  W,
    int                  H
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= W || i >= H) return;

    unsigned char val = 0;
    #pragma unroll
    for (int di = -2; di <= 2; di++) {
        #pragma unroll
        for (int dj = -2; dj <= 2; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < H && nj >= 0 && nj < W)
                val |= in[ni * W + nj];
        }
    }
    out[i * W + j] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. 5×5 이진 침식 (binary erosion)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void erode_5x5_kernel(
    const unsigned char *__restrict__ in,
    unsigned char       *out,
    int                  W,
    int                  H
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= W || i >= H) return;

    unsigned char val = 1;
    #pragma unroll
    for (int di = -2; di <= 2; di++) {
        #pragma unroll
        for (int dj = -2; dj <= 2; dj++) {
            int ni = i + di, nj = j + dj;
            if (ni < 0 || ni >= H || nj < 0 || nj >= W)
                val = 0;
            else
                val &= in[ni * W + nj];
        }
    }
    out[i * W + j] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. 최종 그리드 조립
//
//   priority (낮은 순):
//     free_grid  → 0
//     avoid_grid → 70
//     curb_grid  → 70
//     person_grid→ 88
//   기본값 -1 (unknown)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void merge_final_kernel(
    const unsigned char *__restrict__ free_grid,
    const unsigned char *__restrict__ avoid_grid,
    const unsigned char *__restrict__ curb_grid,
    const unsigned char *__restrict__ person_grid,
    signed char         *grid_np,
    int                  N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    signed char val = -1;
    if (free_grid  [idx]) val =  0;
    if (avoid_grid [idx]) val = 70;
    if (curb_grid  [idx]) val = 80;   // curb: avoid(70)보다 높은 값으로 분리
    if (person_grid[idx]) val = 88;
    grid_np[idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. 카메라 시점 기반 ray gap fill
//
//   sensor 위치에서 그리드 경계 셀까지 Bresenham 광선을 쏜다 (thread 1개 = 광선 1개).
//   광선 위에서 다음 패턴이 감지되면 avoid/curb(70) 셀을 free(0)로 채운다:
//     free(0) → avoid/curb(70, ≤max_gap 셀) → free(0, ≥min_free_after 셀)
//
//   person(88) 셀을 만나면 광선 추적을 즉시 중단한다 (안전).
//   unknown(-1) 셀은 free/obstacle 어느 쪽도 아닌 것으로 처리한다.
//
//   boundary cell 수: 2*(W+H-2)  (corners 중복 제외)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void ray_gap_fill_final_kernel(
    signed char *grid,
    int          sensor_i,
    int          sensor_j,
    int          W,
    int          H,
    int          max_gap,
    int          min_free_after
)
{
    int n_total = 2 * (W + H - 2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    // thread idx → 경계 셀 (end_i, end_j) 매핑
    // top row(W) → right col(H-1) → bottom row(W-1, 오른쪽→왼쪽) → left col(H-2, 아래→위)
    int end_i, end_j;
    int nt = W;
    int nr = H - 1;
    int nb = W - 1;
    if (idx < nt) {
        end_i = 0;     end_j = idx;
    } else if (idx < nt + nr) {
        end_i = idx - nt + 1; end_j = W - 1;
    } else if (idx < nt + nr + nb) {
        end_i = H - 1; end_j = W - 2 - (idx - nt - nr);
    } else {
        end_i = H - 2 - (idx - nt - nr - nb); end_j = 0;
    }

    // Bresenham 파라미터
    int di  = end_i - sensor_i;
    int dj  = end_j - sensor_j;
    int adi = (di < 0) ? -di : di;
    int adj = (dj < 0) ? -dj : dj;
    int si  = (di >= 0) ? 1 : -1;
    int sj  = (dj >= 0) ? 1 : -1;

    int  steps   = (adj > adi) ? adj : adi;
    bool major_j = (adj > adi);
    int  err     = major_j ? (adj / 2) : (adi / 2);

    // gap 버퍼 (max_gap ≤ RAY_GAP_BUF 가정)
#define RAY_GAP_BUF 60
    int gap_ri[RAY_GAP_BUF];
    int gap_rj[RAY_GAP_BUF];
    int  gap_len    = 0;
    bool in_gap     = false;
    bool had_free   = false;
    int  free_after = 0;

    int ci = sensor_i, cj = sensor_j;
    for (int s = 0; s <= steps; s++) {
        if (ci < 0 || ci >= H || cj < 0 || cj >= W) break;

        signed char v = grid[ci * W + cj];
        bool is_free   = (v == 0);
        bool is_avoid  = (v == 70);
        bool is_person = (v == 88);

        if (is_person) break;  // person 뒤로는 절대 채우지 않음

        if (!in_gap) {
            if (is_free) {
                had_free = true;
            } else if (is_avoid && had_free) {
                // gap 진입
                in_gap = true; gap_len = 0; free_after = 0;
                gap_ri[gap_len] = ci; gap_rj[gap_len] = cj; gap_len++;
            }
        } else {
            if (is_avoid) {
                if (gap_len < max_gap && gap_len < RAY_GAP_BUF) {
                    gap_ri[gap_len] = ci; gap_rj[gap_len] = cj; gap_len++;
                } else {
                    // 너무 큰 gap → 실제 장애물로 판단, 포기
                    in_gap = false; gap_len = 0; free_after = 0;
                }
            } else if (is_free) {
                free_after++;
                if (free_after >= min_free_after) {
                    // 패턴 확인 → gap 셀을 free로 채움
                    for (int g = 0; g < gap_len; g++)
                        grid[gap_ri[g] * W + gap_rj[g]] = 0;
                    in_gap = false; gap_len = 0; free_after = 0;
                    // had_free 유지 — 이후 구간에서도 gap 탐지 계속
                }
            } else {
                // unknown(-1) 또는 기타 → gap 포기
                in_gap = false; gap_len = 0; free_after = 0;
            }
        }

        // Bresenham 이동
        if (major_j) {
            cj += sj; err -= adi;
            if (err < 0) { ci += si; err += adj; }
        } else {
            ci += si; err -= adj;
            if (err < 0) { cj += sj; err += adi; }
        }
    }
}



} // extern "C"
