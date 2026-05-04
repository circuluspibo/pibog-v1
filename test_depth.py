import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
import cv2

def main():
    # 1. 경로 및 설정
    model_path = "./models/depth_anything_v2_int8.xml"
    image_path = "furseal.jpg"
    target_height, target_width = 518, 518  # 추론을 고정할 해상도 (518은 모델 권장 크기 중 하나)
    
    # 2. OpenVINO Core 초기화
    core = ov.Core()
    model = core.read_model(model_path)
    
    # 3. 모델 형상 고정 (Reshape) - Dynamic Shape 에러 해결 핵심
    # 입력 레이어의 이름을 가져와서 [배치, 채널, 높이, 너비] 순으로 고정합니다.
    input_layer = model.input(0)
    model.reshape({input_layer.any_name: [1, 3, target_height, target_width]})
    
    # 4. 모델 컴파일 (장치 선택: "CPU", "GPU", "NPU")
    # Red Hat Summit/Computex 데모 시 "GPU"나 "NPU"로 변경하여 테스트 권장
    compiled_model = core.compile_model(model, "CPU")
    output_layer = compiled_model.output(0)
    
    print(f"모델 입력이 [{target_height}x{target_width}]로 고정되었습니다.")

    # 5. 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: {image_path} 파일을 찾을 수 없습니다.")
        return
    
    original_shape = image.shape[:2] # (H, W)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 고정된 모델 입력 크기에 맞춰 리사이즈
    resized_image = cv2.resize(image_rgb, (target_width, target_height))
    
    # HWC(높이,너비,채널) -> CHW(채널,높이,너비) 및 정규화
    input_data = resized_image.transpose(2, 0, 1)
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, 0) # 배치 차원 [1, 3, 518, 518]

    # 6. 추론 실행
    result = compiled_model([input_data])[output_layer]
    depth_map = result[0] # 출력 형태: (518, 518) 혹은 (1, 518, 518)

    # 차원 정리 (필요 시)
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)

    # 7. 후처리 (원본 크기로 복원 및 시각화)
    depth_map_rescaled = cv2.resize(depth_map, (original_shape[1], original_shape[0]))
    
    # 시각화용 정규화 (Min-Max Scaling)
    depth_min = depth_map_rescaled.min()
    depth_max = depth_map_rescaled.max()
    depth_norm = (depth_map_rescaled - depth_min) / (depth_max - depth_min)
    depth_viz = (depth_norm * 255).astype(np.uint8)
    
    # 컬러맵 적용
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)

    # 8. 결과 저장 및 출력
    output_path = "furseal_depth_fixed.jpg"
    cv2.imwrite(output_path, depth_color)
    
    print("-" * 40)
    print(f"결과 저장: {output_path}")
    print(f"출력 데이터 타입: {depth_map.dtype}")
    print(f"최소값: {depth_min:.4f} / 최대값: {depth_max:.4f}")
    print(f"중앙 픽셀 Depth: {depth_map_rescaled[original_shape[0]//2, original_shape[1]//2]:.4f}")
    print("-" * 40)

    # 화면에 결과 표시
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Depth Anything V2 (Fixed Shape)")
    plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()