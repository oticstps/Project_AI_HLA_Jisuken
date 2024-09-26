import cv2
from imutils.video import VideoStream
import time
from ultralytics import YOLO
import os
from datetime import datetime
import mysql.connector
import serial  # Import the pyserial library

# Load YOLO model
model = YOLO("yolov8n.pt")
rtsp_base_url = "rtsp://admin:pt_otics1*@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
common_paths = ["/cam/realmonitor?channel=1&subtype=0"]
class_names = ['hla', 'off', 'altar', 'box_after']
media_folder = "media"

# Serial communication setup
serial_port = "/dev/ttyUSB0"
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)  # Open serial port

if not os.path.exists(media_folder):
    os.makedirs(media_folder)

# MySQL connection details
db_config = {
    "host": "localhost",
    "user": "username",
    "password": "password",
    "database": "db_tps_hla_iaa32"
}

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











def update_db(hla_count):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        andon_value = 1 if hla_count > 0 else 0
        sql_query = "UPDATE tb_andon SET andon = %s WHERE id = %s"
        cursor.execute(sql_query, (andon_value, 1))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Successfully updated andon to {andon_value} for id 1")
    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL Error: {e}")






def capture_image(frame):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = os.path.join(media_folder, f"hla_capture_{current_time}.jpg")
    print(f"[INFO] Capturing image: {image_save_path}")
    cv2.imwrite(image_save_path, frame)
    print(f"[INFO] Image saved to {image_save_path}")



def send_serial(hla_count):
    if hla_count >= 1:
        ser.write(b'8\n')  # Send '8' when hla count reaches 50
        print(f"[INFO] Sent serial data: 8 (HLA count: {hla_count})")
    else:
        ser.write(b'100\n')  # Send '100' for other cases
        print(f"[INFO] Sent serial data: 100 (HLA count: {hla_count})")

last_capture_time = 0
capture_cooldown = 10










while True:
    time.sleep(0.3)
    frame = camera_stream.read()
    if frame is None:
        break
    else:
        try:
            results = model(frame)
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
        send_serial(hla_count)  # Send data via serial
        if (current_time - last_capture_time) >= capture_cooldown:
            update_db(hla_count)
            
            if hla_count == 1:
                capture_image(frame)
            last_capture_time = current_time
        
        annotated_frame = results[0].plot(line_width=2, labels=True, conf=True)
        resized_frame = cv2.resize(annotated_frame, (1395, 770))
        cv2.imshow("Deteksi Part HLA", resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Close the serial port when done
ser.close()

camera_stream.stop()
cv2.destroyAllWindows()
