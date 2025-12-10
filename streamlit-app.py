import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Object Detection Settings")

algorithm_choice = st.sidebar.selectbox(
    "Select Detection Algorithm",
    ["YOLO (v8)", "Detectron2", "SSD (Single Shot Detector)", "Faster R-CNN", "RetinaNet"]
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

input_choice = st.sidebar.radio("Input Source", ["Upload Image", "Upload Video", "Webcam"])

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_model(algorithm):
    if algorithm == "YOLO (v8)":
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    elif algorithm == "SSD (Single Shot Detector)":
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
        model.eval()
        return model
    elif algorithm == "Faster R-CNN":
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model
    elif algorithm == "RetinaNet":
        import torch
        from torchvision.models.detection import retinanet_resnet50_fpn
        model = retinanet_resnet50_fpn(pretrained=True)
        model.eval()
        return model
    elif algorithm == "Detectron2":
        import detectron2
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return predictor

model = load_model(algorithm_choice)

# -----------------------------
# Frame Processing
# -----------------------------
def process_frame(frame, algorithm, model, conf):
    import cv2
    import numpy as np

    if algorithm == "YOLO (v8)":
        results = model(frame, conf=conf)
        annotated = results[0].plot()
        detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

    elif algorithm in ["SSD (Single Shot Detector)", "Faster R-CNN", "RetinaNet"]:
        import torch
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        input_tensor = transform(frame).unsqueeze(0)
        outputs = model(input_tensor)[0]
        detected_objects = [str(label.item()) for label, score in zip(outputs['labels'], outputs['scores']) if score > conf]
        annotated = frame.copy()
        for box, score in zip(outputs['boxes'], outputs['scores']):
            if score > conf:
                x1, y1, x2, y2 = box.int()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    elif algorithm == "Detectron2":
        outputs = model(frame)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        detected_objects = [str(c) for c, s in zip(classes, scores) if s > conf]
        annotated = frame.copy()
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s > conf:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return annotated, detected_objects

# -----------------------------
# Input Handling
# -----------------------------
if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        annotated_img, objects = process_frame(img, algorithm_choice, model, confidence_threshold)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        st.write("Detected Objects:", objects)

elif input_choice == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, objects = process_frame(frame, algorithm_choice, model, confidence_threshold)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

elif input_choice == "Webcam":
    import streamlit_webrtc
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, _ = process_frame(img, algorithm_choice, model, confidence_threshold)
            return annotated_frame

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
