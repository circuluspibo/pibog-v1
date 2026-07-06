import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # 1. 파이프라인 및 설정 초기화
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. 스트림 활성화 (RGB: 640x480, Depth: 640x480, 30fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. 스트리밍 시작
    try:
        pipeline.start(config)
        print("[INFO] RealSense 카메라 스트리밍을 시작합니다. (종료: 'q' 키)")
    except Exception as e:
        print(f"[ERROR] 카메라를 연결할 수 없습니다: {e}")
        return

    try:
        while True:
            # 4. 프레임 세트 대기 및 가져오기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 5. 이미지를 numpy 배열로 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 6. 깊이(Depth) 이미지를 시각화하기 위해 컬러 맵 적용
            # (16비트 깊이 데이터를 8비트 이미지로 변환 후 컬러 적용)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # 7. RGB 이미지와 깊이 이미지를 가로로 병합하여 하나의 창에 표시
            images = np.hstack((color_image, depth_colormap))

            # 8. 화면 출력 및 종료 조건 설정
            cv2.imshow('RealSense Test (Left: RGB / Right: Depth)', images)
            
            # 'q' 키를 누르면 루프 탈출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 9. 스트리밍 종료 및 창 닫기
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 스트리밍이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
