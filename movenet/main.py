# movenet_lightning_tflite.py
# 17 keypoints + skeleton (COCO)
# input: video file

import cv2
import numpy as np
import tensorflow as tf

# ======================
# COCO Skeleton 정의
# ======================
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# ======================
# Keypoint + Skeleton 그리기
# ======================
def draw_pose(frame, keypoints, conf_th=0.2):
    h, w, _ = frame.shape
    keypoints = keypoints[0, 0]  # (17, 3)

    # 점
    for y, x, score in keypoints:
        if score < conf_th:
            continue
        cx, cy = int(x * w), int(y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # 선
    for i, j in SKELETON:
        y1, x1, s1 = keypoints[i]
        y2, x2, s2 = keypoints[j]
        if s1 < conf_th or s2 < conf_th:
            continue
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ======================
# MoveNet Lightning 로드
# ======================
interpreter = tf.lite.Interpreter(
    model_path="movenet_lightning.tflite"
)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ======================
# 비디오 입력
# ======================
cap = cv2.VideoCapture("01_Database/01.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔑 Lightning 전처리 (192x192, int32)
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]["index"])
    draw_pose(frame, keypoints, conf_th=0.2)

    cv2.imshow("MoveNet Lightning (TFLite)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
