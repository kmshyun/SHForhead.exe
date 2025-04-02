import cv2
import mediapipe as mp
import numpy as np

def is_forehead_visible(image, left_x, right_x, top_y, bottom_y):
    # 이마 영역 자르기
    forehead_region = image[top_y:bottom_y, left_x:right_x]
    
    # YCrCb 색 공간으로 변환 (피부색 검출에 유리)
    ycrcb = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2YCrCb)
    
    # 피부색 범위 설정 (일반적인 피부색 범위, 조정 가능)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    
    # 피부색 마스크 적용
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    # 피부색 픽셀 비율 계산
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    
    return skin_ratio > 0.3  # 30% 이상 피부색이면 이마가 보인다고 판단

def detect_forehead_ratio(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 좌우 끝 좌표
            left_x = int(face_landmarks.landmark[234].x * w)
            right_x = int(face_landmarks.landmark[454].x * w)
            face_width = right_x - left_x
            
            # 이마 상단 및 하단 계산
            eyebrow_top = face_landmarks.landmark[10]
            forehead_bottom_y = int(eyebrow_top.y * h)
            forehead_top_y = int(eyebrow_top.y * h - 0.2 * h)
            
            # 이마 영역의 넓이 계산
            forehead_area = face_width * (forehead_bottom_y - forehead_top_y)
            
            # 전체 얼굴 영역의 넓이 계산
            chin_bottom = face_landmarks.landmark[152]
            face_height = abs(chin_bottom.y - eyebrow_top.y) * h
            face_area = face_width * face_height
            
            # 이마 비율 계산
            forehead_ratio = forehead_area / face_area
            
            # 이마가 머리카락으로 가려져 있는지 확인
            if not is_forehead_visible(image, left_x, right_x, forehead_top_y, forehead_bottom_y):
                cv2.putText(image, "FOREHEAD COVERED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image
            
            # 이마를 감싸는 직사각형 그리기
            cv2.rectangle(image, (left_x, forehead_top_y), (right_x, forehead_bottom_y), (0, 255, 0), 2)
            
            ratio_text = f"Forehead Ratio: {forehead_ratio:.2f}"
            cv2.putText(image, ratio_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if forehead_ratio >= 0.4:
                cv2.putText(image, "WIDE FOREHEAD GUY@@@@", (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 2)
    
    return image

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_forehead_ratio(frame)
    cv2.imshow("Forehead Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()