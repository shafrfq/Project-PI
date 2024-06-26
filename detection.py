import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import requests
import logging
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fungsi untuk mengunduh file jika belum ada atau unduhan sebelumnya tidak lengkap
def download_file(url, output_path, expected_size=None):
    if not os.path.exists(output_path) or (expected_size and os.path.getsize(output_path) < expected_size):
        logger.info(f"Downloading {url} to {output_path}...")
        try:
            response = requests.get(url, stream=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"Error downloading {url}: {e}")

# Mengunduh model YOLOv3
@st.cache_resource
def load_yolo():
    os.makedirs('yolov3', exist_ok=True)
    download_file('https://pjreddie.com/media/files/yolov3.weights', 'yolov3/yolov3.weights', 248007048)
    download_file('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 'yolov3/yolov3.cfg')
    download_file('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', 'yolov3/coco.names')

    net = cv2.dnn.readNet('yolov3/yolov3.weights', 'yolov3/yolov3.cfg')
    with open('yolov3/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

# Fungsi untuk deteksi objek
def detect_objects(net, classes, output_layers, image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]*100:.2f}%"
            color = (0, 255, 0)  # Hijau untuk bounding box
            text_color = (255, 255, 255)  # Putih untuk teks
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return image

# VideoTransformerBase subclass for real-time object detection
class YOLOv3VideoTransformer(VideoTransformerBase):
    def __init__(self, net, classes, output_layers):
        self.net = net
        self.classes = classes
        self.output_layers = output_layers

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        detected_image = detect_objects(self.net, self.classes, self.output_layers, image)
        return detected_image

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Object Detection using YOLOv3")
    st.write("Upload an image, video, or use your webcam for object detection")

    net, classes, output_layers = load_yolo()

    option = st.selectbox('Choose an option:', ('Image', 'Video', 'Webcam'))

    if option == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)

            if st.button("Detect Objects"):
                st.write("Detecting...")
                detected_image = detect_objects(net, classes, output_layers, image)
                st.image(detected_image, channels="BGR", caption='Detected Image.', use_column_width=True)

                is_success, buffer = cv2.imencode(".jpg", detected_image)
                if is_success:
                    st.download_button(
                        label="Download Detected Image",
                        data=buffer.tobytes(),
                        file_name="detected_image.jpg",
                        mime="image/jpeg"
                    )

                if st.button("Back to Start"):
                    st.experimental_rerun()

    elif option == 'Video':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if st.button("Detect Objects in Video"):
                st.write("Detecting...")
                output_video_path = detect_video(net, classes, output_layers, video_path)
                st.video(output_video_path)
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Detected Video",
                        data=file,
                        file_name="output_detection.mp4",
                        mime="video/mp4"
                    )

                if st.button("Back to Start"):
                    st.experimental_rerun()

    elif option == 'Webcam':
        st.write("Loading available cameras...")
        # JavaScript code to get the list of video devices
        get_video_devices_js = """
        <script>
        async function getVideoDevices() {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            return videoDevices;
        }
        </script>
        """
        st.components.v1.html(get_video_devices_js, height=0)

        # JavaScript code to send the list of video devices to Streamlit
        st.components.v1.html("""
        <script>
        async function sendVideoDevices() {
            const videoDevices = await getVideoDevices();
            const videoDevicesList = videoDevices.map(device => ({ label: device.label, id: device.deviceId }));
            const jsonVideoDevicesList = JSON.stringify(videoDevicesList);
            const streamlitVideoDevicesElement = document.createElement('streamlit-video-devices');
            streamlitVideoDevicesElement.textContent = jsonVideoDevicesList;
            document.body.appendChild(streamlitVideoDevicesElement);
        }
        sendVideoDevices();
        </script>
        """, height=0)

        # Hidden Streamlit element to receive the list of video devices
        video_devices = st.text_area("Available Video Devices", "", height=1)
        video_devices_list = st.experimental_get_query_params().get('video_devices', [])

        # Parse the video devices list
        import json
        if video_devices_list:
            video_devices = json.loads(video_devices_list[0])
            video_device_labels = [device['label'] for device in video_devices]
        else:
            video_device_labels = []

        camera_option = st.selectbox('Select Camera:', video_device_labels)

        if camera_option:
            selected_device = next(device for device in video_devices if device['label'] == camera_option)

            webrtc_ctx = webrtc_streamer(
                key="example",
                video_transformer_factory=lambda: YOLOv3VideoTransformer(net, classes, output_layers),
                media_stream_constraints={
                    "video": {
                        "deviceId": {
                            "exact": selected_device['id']
                        }
                    },
                    "audio": False
                },
                async_transform=True
            )

            if st.button("Back to Start"):
                st.experimental_rerun()

if __name__ == "__main__":
    main()

st.caption('Copyright (C) Shafira Fimelita Q - 2024')
