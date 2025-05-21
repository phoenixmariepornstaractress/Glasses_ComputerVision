# The code was originally written by TiffinTech and later developed further by Phoenix Marie.

import cv2
import numpy as np
from PIL import Image
import os
import time

# Image overlay with alpha blending
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    overlay_image = img_overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    img_crop = img[y1:y2, x1:x2]

    alpha = overlay_image[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = alpha * overlay_image[:, :, c] + alpha_inv * img_crop[:, :, c]

    img[y1:y2, x1:x2] = img_crop
    return img

# Load transparent glasses image
def load_glasses_image(filename):
    img = Image.open(filename).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

# Rank glasses score based on face shape
def rank_glasses_for_oval_face(glasses_width, face_width, face_height, glasses_index):
    ideal_ratio = 1.5
    face_ratio = face_width / face_height
    glasses_face_ratio = glasses_width / face_width

    width_score = 10 - abs(glasses_face_ratio - 1) * 10
    ratio_score = 10 - abs(face_ratio - ideal_ratio) * 5
    unique_factor = [0.8, 1.0, 1.2, 0.9][glasses_index % 4]

    total_score = (width_score * 0.5 + ratio_score * 0.3 + unique_factor * 2)
    return min(max(total_score, 0), 10)

# Sidebar with glow effect
def create_sidebar(glasses_images, current_glasses, frame_height):
    sidebar = np.zeros((frame_height, 100, 3), dtype=np.uint8)
    thumbnail_height = frame_height // len(glasses_images)
    for i, img in enumerate(glasses_images):
        y = i * thumbnail_height
        resized = cv2.resize(img[:, :, :3], (80, thumbnail_height - 20))
        sidebar[y + 10:y + thumbnail_height - 10, 10:90] = resized
        if i == current_glasses:
            glow_color = (0, 255, 0)
            for thickness in range(1, 5):
                alpha = 0.2 + 0.2 * (5 - thickness)
                overlay = sidebar.copy()
                cv2.rectangle(overlay, (5 - thickness, y + 5 - thickness),
                              (95 + thickness, y + thumbnail_height - 5 + thickness),
                              glow_color, thickness)
                cv2.addWeighted(overlay, alpha, sidebar, 1 - alpha, 0, sidebar)
    return sidebar

# Save snapshot to file
def save_snapshot(frame, filename_prefix="snapshot"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    cv2.imwrite(filename, frame)

# List glasses images in directory
def list_available_glasses_images(directory="."):
    return [f for f in os.listdir(directory) if f.startswith("glasses") and f.endswith(".png")]

# Calculate center point of face
def calculate_face_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

# Toggle overlay state
def toggle_overlay(toggle_state):
    return not toggle_state

# Average face size for multiple faces
def get_average_face_size(faces):
    if not faces:
        return 0, 0
    total_width = sum([w for (_, _, w, _) in faces])
    total_height = sum([h for (_, _, _, h) in faces])
    return total_width // len(faces), total_height // len(faces)

# Draw bounding boxes around faces
def draw_face_outline(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display current glasses name
def display_glasses_name(frame, name):
    cv2.putText(frame, name, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Print face info to console
def log_face_data(faces):
    print("Detected faces:")
    for i, (x, y, w, h) in enumerate(faces):
        center = calculate_face_center(x, y, w, h)
        print(f"Face {i+1}: Location=({x},{y}), Size=({w}x{h}), Center={center}")

# Choose the best-fitting glasses
def suggest_best_glasses(faces, glasses_images):
    if not faces:
        return 0
    x, y, w, h = faces[0]
    scores = [rank_glasses_for_oval_face(img.shape[1], w, h, idx) for idx, img in enumerate(glasses_images)]
    return int(np.argmax(scores))

# Draw basic instructions on screen
def draw_instructions(frame):
    instructions = [
        "Press 'q' to quit.",
        "Press 'n' for next glasses.",
        "Press 's' to save a snapshot.",
        "Press 'o' to toggle overlay.",
        "Press 'b' to auto-pick best glasses."
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 100 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

# Simulated age estimation
def estimate_face_age(frame, faces):
    return [25 for _ in faces]

# Face count status
def detect_multiple_faces_status(faces):
    return "Multiple faces detected" if len(faces) > 1 else "Single face detected"

# Draw status message
def draw_status_message(frame, message):
    cv2.putText(frame, message, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

# Calculate glasses overlay position
def get_glasses_overlay_position(face):
    x, y, w, h = face
    return (x, y + int(h / 4))

# Check if face is centered
def is_face_centered(frame, face):
    frame_center_x = frame.shape[1] // 2
    face_center_x = face[0] + face[2] // 2
    return abs(frame_center_x - face_center_x) < face[2] // 4

# Prompt user to center face
def draw_centering_tip(frame, face):
    if not is_face_centered(frame, face):
        cv2.putText(frame, "Center your face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Save detected face data
def save_face_data(faces, filename="face_data.txt"):
    with open(filename, "a") as f:
        for i, (x, y, w, h) in enumerate(faces):
            center = calculate_face_center(x, y, w, h)
            f.write(f"Face {i+1}: Location=({x},{y}), Size=({w}x{h}), Center={center}\n")

# Main application logic
def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    glasses_filenames = list_available_glasses_images()
    glasses_images = [load_glasses_image(name) for name in glasses_filenames]
    current_glasses = 0
    show_overlay = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_glasses
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > frame.shape[1]:
                clicked_glasses = y // (frame.shape[0] // len(glasses_images))
                if clicked_glasses < len(glasses_images):
                    current_glasses = clicked_glasses

    cv2.namedWindow('Glasses Try-On App')
    cv2.setMouseCallback('Glasses Try-On App', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        sidebar = create_sidebar(glasses_images, current_glasses, frame.shape[0])

        for (x, y, w, h) in faces:
            if show_overlay:
                glasses = cv2.resize(glasses_images[current_glasses], (w, int(h / 3)))
                pos = get_glasses_overlay_position((x, y, w, h))
                frame = overlay_image_alpha(frame, glasses, pos)

            score = rank_glasses_for_oval_face(glasses_images[current_glasses].shape[1], w, h, current_glasses)
            cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, 80), (0, 255, 0), 2)
            cv2.putText(frame, f"Score:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"{score:.1f}/10", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            draw_centering_tip(frame, (x, y, w, h))

        draw_face_outline(frame, faces)
        display_glasses_name(frame, glasses_filenames[current_glasses])
        draw_instructions(frame)
        log_face_data(faces)
        save_face_data(faces)

        status_msg = detect_multiple_faces_status(faces)
        draw_status_message(frame, status_msg)

        combined_frame = np.hstack((frame, sidebar))
        cv2.imshow('Glasses Try-On App', combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_glasses = (current_glasses + 1) % len(glasses_images)
        elif key == ord('s'):
            save_snapshot(frame)
        elif key == ord('o'):
            show_overlay = toggle_overlay(show_overlay)
        elif key == ord('b'):
            current_glasses = suggest_best_glasses(faces, glasses_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
