import cv2
from imutils.video import VideoStream
import time
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLO model
model = YOLO("best.pt")
rtsp_base_url = "rtsp://admin:pt_otics1*@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
common_paths = ["/cam/realmonitor?channel=1&subtype=0"]
class_names = ['hla', 'off', 'altar', 'box_after']
media_folder = "media"

if not os.path.exists(media_folder):
    os.makedirs(media_folder)

def test_stream(path):
    rtsp_url = rtsp_base_url + path
    print(f"[INFO] testing {rtsp_url}")
    vs = VideoStream(rtsp_url).start()
    time.sleep(2.0)
    frame = vs.read()
    vs.stop()
    return frame is not None

for path in common_paths:
    if test_stream(path):
        print(f"[INFO] stream path found: {path}")
        rtsp_url = rtsp_base_url + path
        break
else:
    print("[ERROR] no valid stream path found")
    exit()

print("[INFO] starting video stream...")
camera_stream = VideoStream(rtsp_url).start()
time.sleep(2.0)

# Capture and save image
def capture_image(frame):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = os.path.join(media_folder, f"hla_capture_{current_time}.jpg")
    print(f"[INFO] Capturing image: {image_save_path}")
    cv2.imwrite(image_save_path, frame)
    print(f"[INFO] Image saved to {image_save_path}")

# List to store timestamp and HLA count history for HLA count 88
capture_history = []

# Capture cooldown
last_capture_time = 0
capture_cooldown = 10

while True:
    time.sleep(0.1)
    frame = camera_stream.read()
    if frame is None:
        break
    else:
        try:
            results = model(frame, conf=0.6)
        except Exception as e:
            print(f"[ERROR] Error during model inference: {e}")
            continue
        
        detected_objects = results[0].boxes.data.cpu().numpy()
        hla_count = 0
        for obj in detected_objects:
            class_id = int(obj[5])
            if class_id < len(class_names):
                class_name = class_names[class_id]
                if class_name == 'hla':
                    hla_count += 1
            else:
                print(f"[WARNING] Detected class_id {class_id} out of bounds for class_names")

        print(f"Number of 'hla' objects detected: {hla_count}")
        current_time = time.time()
        if (current_time - last_capture_time) >= capture_cooldown:
            # Capture image and store timestamp and HLA count when hla_count reaches 1
            if hla_count == 88:
                capture_image(frame)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                capture_history.append((timestamp, hla_count))  # Store timestamp and HLA count as a tuple
                # Keep the history list to a maximum of 30 entries for display
                if len(capture_history) > 30:
                    capture_history.pop(0)
            last_capture_time = current_time
        
        # Annotate the frame with the detection results
        annotated_frame = results[0].plot(line_width=1, labels=True, conf=True)
        
        # Create a dark blue sidebar on the left side for the 'HLA' count display
        sidebar_width = 500  # Width of the sidebar
        sidebar_color = (142, 112, 0)  # Dark blue color (BGR format)
        cv2.rectangle(annotated_frame, (0, 0), (sidebar_width, annotated_frame.shape[0]), sidebar_color, -1)  # Filled dark blue sidebar

        # Add a red background for the 'HLA Count' display
        red_bg_color = (0, 0, 255)  # Red color (BGR format)
        cv2.rectangle(annotated_frame, (20, 30), (sidebar_width - 20, 100), red_bg_color, -1)  # Filled red background

        # Overlay the 'HLA Count' text on the red background with bold white text
        white_text_color = (255, 255, 255)  # White color for text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Overlay the 'HLA Count' text
        cv2.putText(annotated_frame, f"HLA Count: {hla_count}", (30, 80), font, 1.5, white_text_color, 5)

        # Display the history of timestamps and HLA counts for HLA count 88 in the sidebar
        history_start_y = 150  # Start Y position for history display
        for idx, (timestamp, count) in enumerate(capture_history):
            history_text = f"{idx + 1}: {timestamp} | HLA: {count}"
            cv2.putText(annotated_frame, history_text, (30, history_start_y + (idx * 30)), font, 0.7, white_text_color, 2)

        # Add a border/frame around the entire display
        frame_color = (184, 132, 0)  # Green frame color (BGR format)
        frame_thickness = 20  # Thickness of the frame
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1] - 1, annotated_frame.shape[0] - 1), frame_color, frame_thickness)

        # Resize the frame for display
        resized_frame = cv2.resize(annotated_frame, (1395, 770))
        
        # Show the frame in a window
        cv2.imshow("Deteksi Part HLA", resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera_stream.stop()
cv2.destroyAllWindows()
