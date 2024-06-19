# CCTV Motion Detection and Notification Bot

This project is a CCTV motion detection system that uses YOLO for object detection. When motion is detected, it captures an image, highlights the detected objects, and sends the image to a specified Telegram chat.

### Prerequisites

1. Python 3.6+
2. OpenCV
3. NumPy
4. Requests
5. Python-dotenv

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/codeirawan/yolo-cctv-bot.git
cd yolo-cctv-bot
```

### 2. Install Dependencies

Make sure you have `pip` installed. Then run:

```bash
pip install opencv-python-headless numpy requests python-dotenv
```

### 3. Download YOLO Files

Download the following files and place them in the project directory:

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true)
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true)

### 4. Create .env File

Create a .env file in the root of the project directory by copying the .env.example file and filling in your Telegram bot token and chat ID:

```
cp .env.example .env
```

Then edit the .env file:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_chat_id
```

## Running the Script

### 1. Ensure Your Webcam is Connected

This script uses the first webcam connected to your computer. Ensure your webcam is properly connected and functional.

### 2. Run the Script

```bash
python yolo.py
```

## How it Works

1. **Motion Detection**: Uses OpenCV's background subtraction to detect motion in the video feed.
2. **Object Detection**: Applies YOLO to detect objects in frames where motion is detected.
3. **Notification**: Captures the frame, highlights detected objects, and sends the image to a Telegram chat.

## Customization

- Adjust the confidence threshold for object detection in the script as needed.
- Modify the object detection classes by updating the `coco.names` file.

## Troubleshooting

- If you encounter issues with the webcam not opening, ensure that your webcam drivers are up to date.
- Check your internet connection if the Telegram notifications are not being sent.

## License

This project is licensed under the MIT License.

```

### Example Tree Structure

```plaintext
yolo-cctv-bot/
├── yolov3.weights
├── yolov3.cfg
├── coco.names
├── yolo.py
├── .env
├── images/
└── README.md
```

### Notes:

- Adjust the area threshold and confidence threshold based on your needs.
- The script uses `MOG2` for motion detection and YOLO for object detection.
- Captured images are saved in the `images` directory and sent to a Telegram chat with labels and red borders highlighting detected objects.
