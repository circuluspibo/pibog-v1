from ultralytics import FastSAM
import cv2
import numpy as np
# Define an inference source
source = "road.jpg"
# Create a FastSAM model
model = FastSAM("./FastSAM-x_int8_openvino_model")  # or FastSAM-x.pt
results = model(source, device="CPU",retina_masks=True, imgsz=1024, conf=0.8, iou=0.9, texts=['road']) #'street','asphalt'


#results[0].plot("mask_result.jpg") 

#print(results[0])
result = results[0]
img = cv2.imread(source)
masks = result.masks.data.cpu().numpy()  # (N, H, W)

print(result)

# 복사본 만들기 (출력용)
overlay = img.copy()

# 마스크 색상 및 투명도 설정
mask_color = (0, 255, 0)  # 녹색 (BGR 형식)
alpha = 0.5               # 투명도

for mask in masks:
    colored_mask = (mask * 255).astype(np.uint8)

    # 컬러 마스크 생성
    color_layer = np.zeros_like(img, dtype=np.uint8)
    color_layer[:, :] = mask_color

    # 마스크 적용
    mask_3ch = cv2.merge([colored_mask] * 3)
    masked_color = cv2.bitwise_and(color_layer, mask_3ch)

    # 오버레이에 컬러 마스크 반영
    overlay = np.where(mask_3ch > 0, cv2.addWeighted(overlay, 1 - alpha, masked_color, alpha, 0), overlay)

    # 윤곽선 그리기 (선택 사항)
    contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # 빨간 윤곽선

# 결과 저장
cv2.imwrite("./road_result.jpg", overlay)


# 바운딩 박스 그리기
"""
for box in result.boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
"""
# 저장