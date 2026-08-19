import cv2
import numpy as np
import mediapipe as mp

def classify_body_shape(shoulder_cm, waist_cm, hip_cm):
    """
    Phân loại dáng người dựa trên tỷ lệ chiều rộng bề ngang Vai, Eo, Hông (2D width ratios).
    """
    if shoulder_cm <= 0 or waist_cm <= 0 or hip_cm <= 0:
        return "Uncertain", "Không đủ dữ liệu để xác định."

    max_sh_hip = max(shoulder_cm, hip_cm)
    min_sh_hip = min(shoulder_cm, hip_cm)
    sh_hip_diff_ratio = abs(shoulder_cm - hip_cm) / max_sh_hip

    # 1. Dáng Quả táo (Apple): Eo to gần bằng hoặc lớn hơn Vai/Hông
    if (waist_cm / shoulder_cm >= 0.85) or (waist_cm / hip_cm >= 0.85):
        return "Apple (Dáng Quả Táo)", "Phần thân trên và eo đầy đặn. Gợi ý: Trang phục cổ chữ V, đầm dáng xòe A-line nhẹ, tránh thắt lưng quá chặt."

    # 2. Dáng Tam giác ngược (Inverted Triangle): Vai rộng hơn Hông rõ rệt (trên 5%)
    elif (shoulder_cm / hip_cm) >= 1.05:
        return "Inverted Triangle (Tam Giác Ngược)", "Vai rộng, hông hẹp. Gợi ý: Quần ống rộng, chân váy xòe, áo đơn giản tối màu ở phần trên."

    # 3. Dáng Quả lê (Pear): Hông rộng hơn Vai rõ rệt (trên 5%)
    elif (hip_cm / shoulder_cm) >= 1.05:
        return "Pear (Dáng Quả Lê)", "Hông nở, vai nhỏ. Gợi ý: Áo có bèo nhún/đệm vai, quần hoặc chân váy suông thẳng tối màu."

    # 4. Vai và Hông xấp xỉ bằng nhau (Chênh lệch dưới 5%)
    else:
        # Đường thắt eo rõ rệt (Eo nhỏ hơn ít nhất 25% so với Vai/Hông)
        if (waist_cm / min_sh_hip) <= 0.75:
            return "Hourglass (Dáng Đồng Hồ Cát)", "Tỷ lệ chuẩn với đường thắt eo rõ. Gợi ý: Đầm ôm sát, áo chiết eo, thắt lưng tôn dáng."
        else:
            return "Rectangle (Dáng Chữ Nhật)", "Thân hình thẳng, đường thắt eo ít rõ. Gợi ý: Tạo điểm nhấn eo bằng thắt lưng, chân váy xếp ly, đầm xòe."


def measure_body_ratios(image_path, user_height_cm=170.0, output_path="result.jpg"):
    mp_pose = mp.solutions.pose
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể tải ảnh. Vui lòng kiểm tra lại đường dẫn!")
        
    h, w, _ = image.shape

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.6
    ) as pose:
        
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            print("Không phát hiện được người trong ảnh!")
            return None

        # 1. Trích xuất mặt nạ người & Tính chiều cao Pixel
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        y_indices = np.where(mask > 0)[0]
        
        if len(y_indices) == 0:
            return None
            
        top_y = np.min(y_indices)
        bottom_y = np.max(y_indices)
        total_height_px = bottom_y - top_y
        
        # Hệ số quy đổi Pixel -> CM
        px_to_cm = user_height_cm / total_height_px if total_height_px > 0 else 0

        # Tọa độ khớp
        landmarks = results.pose_landmarks.landmark
        l_shoulder = np.array([int(landmarks[11].x * w), int(landmarks[11].y * h)])
        r_shoulder = np.array([int(landmarks[12].x * w), int(landmarks[12].y * h)])
        l_hip = np.array([int(landmarks[23].x * w), int(landmarks[23].y * h)])
        r_hip = np.array([int(landmarks[24].x * w), int(landmarks[24].y * h)])

        def get_body_width_at_y(y_coord):
            if y_coord < 0 or y_coord >= h:
                return 0, 0, 0
            row = mask[y_coord, :]
            nonzero_indices = np.where(row > 0)[0]
            if len(nonzero_indices) > 1:
                return (nonzero_indices[-1] - nonzero_indices[0]), nonzero_indices[0], nonzero_indices[-1]
            return 0, 0, 0

        # --- ĐO KÍCH THƯỚC (VAI, EO, HÔNG) ---
        shoulder_width_px = int(np.linalg.norm(l_shoulder - r_shoulder))
        shoulder_width_cm = round(shoulder_width_px * px_to_cm, 1)
        y_shoulder_avg = int((l_shoulder[1] + r_shoulder[1]) / 2)

        y_hip_avg = int((l_hip[1] + r_hip[1]) / 2)
        y_waist_start = y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.40)
        y_waist_end = y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80)

        min_waist_px = float('inf')
        waist_y, waist_x1, waist_x2 = -1, 0, 0

        for y in range(y_waist_start, y_waist_end):
            width, x1, x2 = get_body_width_at_y(y)
            if 0 < width < min_waist_px:
                min_waist_px = width
                waist_y, waist_x1, waist_x2 = y, x1, x2

        waist_cm = round(min_waist_px * px_to_cm, 1) if min_waist_px != float('inf') else 0.0

        y_hip_search_start = y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80)
        y_hip_search_end = min(h - 1, y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 1.25))

        max_hip_px = 0
        hip_y, hip_x1, hip_x2 = -1, 0, 0

        for y in range(y_hip_search_start, y_hip_search_end):
            width, x1, x2 = get_body_width_at_y(y)
            if width > max_hip_px:
                max_hip_px = width
                hip_y, hip_x1, hip_x2 = y, x1, x2

        hip_cm = round(max_hip_px * px_to_cm, 1)

        # --- PHÂN LOẠI DÁNG NGƯỜI ---
        body_shape_name, styling_advice = classify_body_shape(shoulder_width_cm, waist_cm, hip_cm)

        # --- VẼ LÊN ẢNH RESULT ---
        annotated_image = image.copy()

        # Đường Vai, Eo, Hông
        cv2.line(annotated_image, tuple(l_shoulder), tuple(r_shoulder), (0, 255, 0), 3)
        cv2.putText(annotated_image, f"Vai: {shoulder_width_cm}cm", (l_shoulder[0] - 100, l_shoulder[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if waist_y != -1:
            cv2.line(annotated_image, (waist_x1, waist_y), (waist_x2, waist_y), (255, 0, 0), 3)
            cv2.putText(annotated_image, f"Eo: {waist_cm}cm", (waist_x1 - 100, waist_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if hip_y != -1:
            cv2.line(annotated_image, (hip_x1, hip_y), (hip_x2, hip_y), (0, 0, 255), 3)
            cv2.putText(annotated_image, f"Hong: {hip_cm}cm", (hip_x1 - 100, hip_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Hiển thị tên dáng người lên góc trên ảnh
        cv2.putText(annotated_image, f"Dang nguoi: {body_shape_name}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite(output_path, annotated_image)

        return {
            "height_cm": user_height_cm,
            "shoulder_cm": shoulder_width_cm,
            "waist_cm": waist_cm,
            "hip_cm": hip_cm,
            "body_shape": body_shape_name,
            "advice": styling_advice
        }

# --- CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    result = measure_body_ratios("full_body.jpg", user_height_cm=165.0, output_path="result_shape.jpg")
    if result:
        print("=== KẾT QUẢ PHÂN TÍCH CHỈ SỐ VÀ DÁNG NGƯỜI ===")
        print(f"- Chiều cao: {result['height_cm']} cm")
        print(f"- Vai: {result['shoulder_cm']} cm | Eo: {result['waist_cm']} cm | Hông: {result['hip_cm']} cm")
        print(f"- Dáng người phân loại: {result['body_shape']}")
        print(f"- Lời khuyên mặc đồ: {result['advice']}")
