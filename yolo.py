import cv2
import numpy as np
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Token and chat ID for Telegram bot from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# Function to send text message to Telegram
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload, timeout=10)  # Timeout set to 10 seconds
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")
        return None

# Function to send photo to Telegram
def send_telegram_photo(photo_path, caption=""):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
    files = {'photo': open(photo_path, 'rb')}
    payload = {
        'chat_id': CHAT_ID,
        'caption': caption
    }
    try:
        response = requests.post(url, files=files, data=payload, timeout=10)
        response.raise_for_status()
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error sending photo: {e}")
        return None

# Create a directory to save captured images
if not os.path.exists('images'):
    os.makedirs('images')

# Load YOLO
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Get the camera source from the environment variable
source = os.getenv('CAMERA_SOURCE')
if source.isdigit():
    source = int(source)

# Initialize webcam capture
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source {source}.")
    exit()

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Apply background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust the area threshold as needed
            motion_detected = True
            break

    if not motion_detected:
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    detected_object = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Adjust this threshold based on your needs
            if confidence > 0.5:
                detected_object = True

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if detected_object:
        # Create a copy of the frame for masking
        mask_frame = frame.copy()

        # Draw rectangles with red border and label on detected objects
        for i in range(len(boxes)):
            if confidences[i] > 0.5:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(mask_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red border
                cv2.putText(mask_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Label

        # Save the frame with detected object to a timestamped file in images folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f'images/detected_object_{timestamp}.jpg'
        cv2.imwrite(image_name, mask_frame)

        # Send photo to Telegram with a caption
        class_names = [classes[class_id] for class_id in class_ids]
        caption = f"Objects detected: {', '.join(class_names)}\nDetected at {timestamp}!"
        send_telegram_photo(image_name, caption)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
