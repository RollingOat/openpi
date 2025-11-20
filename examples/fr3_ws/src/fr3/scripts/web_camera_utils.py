import cv2

def start_web_camera(camera_index=2):
    """Starts the web camera capture."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with index {camera_index}")
    return cap

def get_images_from_web_camera(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera")
    return frame

# def get_images_from_web_camera(camera_index=2):
#     """Captures a single image from the specified web camera."""
#     cap = cv2.VideoCapture(camera_index)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open camera with index {camera_index}")

#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Failed to capture image from camera")

#     return frame


def main():
    cap = cv2.VideoCapture(2)  # 0 = first camera; try 1,2... if you have multiple
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        while True:
            ret, frame = cap.read()   # frame is a numpy array (H, W, BGR)
            if not ret:
                print("Failed to grab frame")
                break
            # print(f"Frame shape: {frame.shape}")
            cv2.imshow("Webcam", frame)
            # press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()