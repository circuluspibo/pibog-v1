import openvino.runtime as ov
import numpy as np
import cv2
import torch
import face_align
import os
from insightface.app import FaceAnalysis

# 1. 전후처리 유틸리티
class ImageUtils:
    def __init__(self, emap_path="emap.npy"):
        if not os.path.exists(emap_path):
            raise FileNotFoundError(f"{emap_path} 파일이 없습니다.")
        self.emap = np.load(emap_path)
        self.input_std = 255.0
        self.input_mean = 0.0

    def postprocess_face(self, face_tensor):
        # 1. 텐서를 넘파이 이미지로 변환
        if isinstance(face_tensor, torch.Tensor):
            face_np = face_tensor.squeeze().cpu().detach().permute(1, 2, 0).numpy()
        else:
            face_np = np.squeeze(face_tensor).transpose(1, 2, 0)
        
        # 2. 0~255 스케일링
        face_np = (face_np * 255).clip(0, 255).astype(np.uint8)
        
        # --- [잡티 제거 블러 로직 추가] ---
        # d=5: 필터링에 사용할 이웃 픽셀의 지름
        # sigmaColor=75: 색상 차이가 이 값보다 작으면 뭉갭니다 (피부 잡티 제거용)
        # sigmaSpace=75: 거리 차이가 이 값보다 작으면 영향을 줍니다
        face_np = cv2.bilateralFilter(face_np, d=12, sigmaColor=50, sigmaSpace=50)
        
        # 추가로 아주 미세한 가우시안 블러를 섞고 싶다면 (선택 사항)
        # face_np = cv2.GaussianBlur(face_np, (3, 3), 0)
        # ----------------------------------

        face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        return face_np

    """
    def postprocess_face(self, face_tensor):
        if isinstance(face_tensor, torch.Tensor):
            face_np = face_tensor.squeeze().cpu().detach().permute(1, 2, 0).numpy()
        else:
            face_np = np.squeeze(face_tensor).transpose(1, 2, 0)
        face_np = (face_np * 255).clip(0, 255).astype(np.uint8)
        face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        return face_np
    """

    def getBlob(self, aimg, input_size=(256, 256)):
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, input_size,
                                    (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        return blob

    def blend_swapped_image(self, swapped_face, target_image, M):
        h, w = target_image.shape[:2]
        M_inv = cv2.invertAffineTransform(M)
        warped_face = cv2.warpAffine(swapped_face, M_inv, (w, h), borderValue=0.0)
        img_white = np.full((swapped_face.shape[0], swapped_face.shape[1]), 255, dtype=np.float32)
        img_mask = cv2.warpAffine(img_white, M_inv, (w, h), borderValue=0.0)
        
        img_mask[img_mask > 20] = 255
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
            mask_h, mask_w = np.max(mask_h_inds) - np.min(mask_h_inds), np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 10, 10)
            img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)
            k = max(mask_size // 20, 5)
            img_mask = cv2.GaussianBlur(img_mask, (k * 2 + 1, k * 2 + 1), 0)
            
        img_mask = np.reshape(img_mask / 255.0, [img_mask.shape[0], img_mask.shape[1], 1])
        result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
        return result.astype(np.uint8)

# 2. 메인 클래스 (GPU 추론 최적화)
class FaceSwapOpenVINO:
    def __init__(self, device_name="GPU", face_enhance=None):
        # device_name을 "GPU"로 설정하여 내장/외장 그래픽 가속 사용
        self.face_enhance = face_enhance
        self.img_utils = ImageUtils(emap_path="emap.npy")
        
        # [A] InsightFace 로드 (onnxruntime-openvino 가속)
        # GPU_FP16 또는 GPU_FP32 장치 지정 가능
        print(f"InsightFace 로딩 중 (가속기: {device_name})...")
        providers = [
            ('OpenVINOExecutionProvider', {
                'device_type': device_name,
                'precision': 'FP16'
            }),
            'CPUExecutionProvider'
        ]
        self.face_detect = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_detect.prepare(ctx_id=0, det_size=(640, 640))
        
        # [B] Reswapper OpenVINO 모델 로드
        print(f"Reswapper 로딩 중 (가속기: {device_name})...")
        self.core = ov.Core()
        # 모델 파일 이름 확인 (앞서 생성한 INT8 모델)
        swap_model_path = "./models/reswapper_256_int8_v3.xml" 
        model_ir = self.core.read_model(swap_model_path)
        
        # GPU 가속을 위해 compile_model 시 device_name 적용
        self.compiled_swap_model = self.core.compile_model(model_ir, device_name)
        self.swap_output_layer = self.compiled_swap_model.output(0)

    def swap_face(self, target_img, source_img, enhance=0):
        target_faces = self.face_detect.get(target_img)
        source_faces = self.face_detect.get(source_img)

        if not target_faces or not source_faces:
            return target_img

        target_face = target_faces[0]
        source_face = source_faces[0]

        aligned_t, M = face_align.norm_crop2(target_img, target_face.kps, 256)
        target_blob = self.img_utils.getBlob(aligned_t, (256, 256))
        
        # Latent 추출
        latent = np.dot(source_face.normed_embedding.reshape((1, -1)), self.img_utils.emap)
        latent /= np.linalg.norm(latent)

        # GPU 추론 실행
        results = self.compiled_swap_model([target_blob, latent.astype(np.float32)])
        swapped_feat = results[self.swap_output_layer]

        swapped_face = self.img_utils.postprocess_face(swapped_feat)
        result_img = self.img_utils.blend_swapped_image(swapped_face, target_img, M)

        if int(enhance) > 0 and self.face_enhance is not None:
            _, _, result_img = self.face_enhance.enhance(
                result_img, has_aligned=False, only_center_face=False, paste_back=True)

        return result_img

# --- 메인 실행부 (imshow 제거) ---
if __name__ == "__main__":
    # GPU로 설정 (Intel 내장 그래픽이 있다면 "GPU" 혹은 "GPU.0")
    # 인식되지 않을 경우 "AUTO"로 설정하면 가장 빠른 장치를 찾습니다.
    swapper = FaceSwapOpenVINO(device_name="GPU")
    
    t_img = cv2.imread("IMAGE1.jpg")
    s_img = cv2.imread("IMAGE2.jpg")
    
    if t_img is not None and s_img is not None:
        print("추론 시작...")
        output = swapper.swap_face(t_img, s_img, enhance=0)
        
        # 결과 저장만 수행
        cv2.imwrite("final_output.jpg", output)
        print("저장 완료: final_output.jpg")
    else:
        print("이미지를 불러오지 못했습니다. 경로를 확인하세요.")