import cv2
import numpy as np
import torch
# 위에서 정의한 FaceSwapOpenVINO 클래스가 포함된 파일에서 로드하거나 같은 파일에 배치하세요.
from ov_faceswap import FaceSwapOpenVINO 

def main():
    # 1. 초기화 (CPU 사용, 필요 시 'GPU'로 변경 가능)
    print("모델 초기화 중...")
    try:
        # face_enhance는 속도를 위해 None으로 설정하거나, 
        # 사용 시 해당 객체를 생성해서 넘겨주세요.
        swapper = FaceSwapOpenVINO(device_name="GPU") 
    except Exception as e:
        print(f"초기화 실패: {e}")
        return

    # 2. 이미지 로드
    # IMAGE1: 얼굴이 바뀔 원본 배경 이미지
    # IMAGE2: 닮고 싶은 대상(스타일/특징 추출용)
    target_img_path = "IMAGE1.JPG" 
    source_img_path = "IMAGE2.JPG"

    target_img = cv2.imread(target_img_path)
    source_img = cv2.imread(source_img_path)

    if target_img is None or source_img is None:
        print("이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"치환 시작: {target_img_path} <--- {source_img_path}")

    # 3. 페이스스왑 실행
    try:
        # enhance=0 (꺼짐), 1 이상이면 인핸서 동작
        result_img = swapper.swap_face(target_img, source_img, enhance=0)
        
        # 4. 결과 출력 및 저장
        #cv2.imshow("Original Target", target_img)
        #cv2.imshow("Source Face", source_img)
        #cv2.imshow("Swapped Result", result_img)
        
        # 이미지 저장
        output_path = "swapped_result4.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"결과가 저장되었습니다: {output_path}")
        
        #print("화면의 창을 닫으려면 아무 키나 누르세요.")
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    except Exception as e:
        print(f"작업 중 오류 발생: {e}")

if __name__ == "__main__":
    main()