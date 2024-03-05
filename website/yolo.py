import argparse
import io
import os
import tempfile
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, Response
from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
import torch
import yaml

yolo = Blueprint('yolo', __name__)

# Get the directory of the CATSWEBREAL directory
catstrafficsign_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load your YOLOv5 models
weights_path_custom = os.path.join(catstrafficsign_dir, 'last.pt')
weights_path_default = os.path.join(catstrafficsign_dir, 'yolov5s.pt')

model_custom = torch.hub.load(os.path.join(catstrafficsign_dir, 'website', 'Yolov5'), 'custom', path=weights_path_custom, source='local')
model_default = torch.hub.load(os.path.join(catstrafficsign_dir, 'website', 'Yolov5'), 'yolov5s', source='local')

model_custom.eval()
model_default.eval()

# Load class names from data.yaml
data_yaml_path = os.path.join(catstrafficsign_dir, 'website', 'Yolov5', 'data.yaml')
with open(data_yaml_path, 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    class_names = data['names']

# Set the class names in the models
model_custom.names = class_names
model_default.names = class_names

def perform_object_detection_on_frame(frame):
    results_custom = model_custom(frame)  # Perform inference using custom YOLOv5 model
    results_default = model_default(frame)  # Perform inference using default YOLOv5 model
    return results_custom, results_default

def annotate_frame(frame, results_custom, results_default):
    # Annotate frame based on custom YOLOv5 results
    for result in results_custom.xyxy[0]:
        class_id = result[-1].int()
        confidence = result[4].item()
        bbox = result[:4].int().tolist()

        if confidence >= 0.5:
            class_name = model_custom.names[class_id]
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Annotate frame based on default YOLOv5 results, appending classes at the end
    allowed_classes = [0, 1, 2, 3, 5, 7, 15, 16]
    for result in results_default.xyxy[0]:
        class_id = result[-1].int()
        confidence = result[4].item()
        bbox = result[:4].int().tolist()

        if confidence >= 0.5 and class_id in allowed_classes:
            # Modify the class name for specific class IDs from the default model
            if class_id == 0:
                class_name = "Person"
            elif class_id == 1:
                class_name = "Bicycle"
            elif class_id == 2:
                class_name = "Car"
            elif class_id == 3:
                class_name = "Motorcycle"
            elif class_id == 5:
                class_name = "Bus"
            elif class_id == 7:
                class_name = "Truck"
            elif class_id == 15:
                class_name = "Cat"
            elif class_id == 16:
                class_name = "Dog"
            else:
                class_name = f"Default_{model_default.names[class_id]}"
            
            # Increment the class_id to avoid overlapping with custom model classes
            class_id += len(model_custom.names)
            
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

@yolo.route("/", methods=["GET", "POST"])
@login_required

def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")
        
        file = request.files["file"]
        if not file:
            return render_template("index.html", message="No file uploaded")

        file_ext = file.filename.rsplit('.', 1)[1].lower()
        if file_ext in ['mp4', 'avi', 'mov', 'mkv']:   # Check if it's a video file
            # Handle video upload and processing (as in your original code)

            video_bytes = file.read()
            temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
            
            with open(temp_video_path, 'wb') as temp_video_file:
                temp_video_file.write(video_bytes)
            
            cap = cv2.VideoCapture(temp_video_path)
            out = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_custom, processed_default = perform_object_detection_on_frame(frame)
                annotated_frame = annotate_frame(frame, processed_custom, processed_default)

                if out is None:
                    frame_height, frame_width, _ = annotated_frame.shape
                    output_video_path = os.path.join(tempfile.gettempdir(), 'processed_output.mp4')
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

                out.write(annotated_frame)
            
            if out is not None:
                out.release()
                return send_file(output_video_path, as_attachment=True)
            else:
                return "Video processing failed."
        
        else:
            img_bytes = file.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
            processed_custom, processed_default = perform_object_detection_on_frame(img)
            annotated_img = annotate_frame(img, processed_custom, processed_default)
            
            output_img_path = os.path.join(tempfile.gettempdir(), 'processed_output.png')
            cv2.imwrite(output_img_path, annotated_img)
            return send_file(output_img_path, as_attachment=True)

    return render_template("index.html")


def gen_frames():
    cap = cv2.VideoCapture(0)  # Access the default webcam (change index for different cameras)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection using both custom and default YOLOv5 models
        results_custom, results_default = perform_object_detection_on_frame(frame)
        
        # Annotate frame based on custom YOLOv5 results
        annotated_frame = annotate_frame(frame, results_custom, results_default)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@yolo.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@yolo.route("/faq")
def faq():
    return render_template("faq.html")

@yolo.route("/about_us")
def about_us():
    return render_template("about_us.html")