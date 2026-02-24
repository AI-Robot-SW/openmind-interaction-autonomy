import time

import cv2
import numpy as np

from providers.realsense_camera_provider import RealSenseCameraProvider


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    # 16-bit depth를 보기 좋게 8-bit로 정규화
    depth = depth.astype(np.float32)
    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    return (depth * 255.0).astype(np.uint8)


def main():
    provider = RealSenseCameraProvider(
        camera_index=0,
        width=640,
        height=480,
        fps=30,
        align_depth_to_color=True,
    )

    provider.start()
    time.sleep(0.3)  # 초기 프레임 채우기

    try:
        while True:
            d = provider.data
            if d is None or d.get("rgb") is None or d.get("depth") is None:
                cv2.waitKey(1)
                continue

            rgb = d["rgb"]
            depth = d["depth"]

            # rgb, depth가 dict로 들어오는 구현도 있고 ndarray로 바로 들어오는 구현도 있어서 둘 다 대응
            if isinstance(rgb, dict):
                rgb_img = rgb.get("image", None)
            else:
                rgb_img = rgb

            if isinstance(depth, dict):
                depth_img = depth.get("image", None)
            else:
                depth_img = depth

            if rgb_img is None or depth_img is None:
                cv2.waitKey(1)
                continue

            # RGB 표시 (OpenCV는 BGR이 기본이라 필요하면 변환)
            if rgb_img.ndim == 3 and rgb_img.shape[2] == 3:
                # rgb_img가 RGB인지 BGR인지 프로젝트 구현에 따라 달라서
                # 색이 이상하면 아래 줄을 반대로 
                bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            else:
                bgr = rgb_img

            # Depth 컬러맵 표시
            depth8 = normalize_depth(depth_img)
            depth_color = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)

            # 간단한 텍스트 오버레이
            ts = d.get("timestamp", None)
            if ts is not None:
                cv2.putText(bgr, f"timestamp: {ts:.3f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("RealSense RGB (provider)", bgr)
            cv2.imshow("RealSense Depth (provider)", depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        provider.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
