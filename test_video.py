import coremltools as ct
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the Core ML model
model = ct.models.MLModel('./runs/train/exp/weights/best.mlpackage')

# Open the video file
video_path = './height120.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object to save the output
output_path = './height120_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL image and resize to match model input size
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_image_resized = pil_image.resize((640, 640))  # Resize the image to model input size

    # Create input dictionary using the PIL image directly
    input_dict = {'image': pil_image_resized}

    # Perform inference
    result = model.predict(input_dict)
    coordinates = result['coordinates']
    confidence = result['confidence']

    # Initialize variables to track the best detection for each class
    best_puck = None
    best_stick = None
    best_puck_confidence = 0
    best_stick_confidence = 0

    # Find the highest confidence for each class
    for i, (bbox, conf) in enumerate(zip(coordinates, confidence)):
        puck_conf, stick_conf = conf  # assuming class 0: puck, class 1: stick

        # Check if this is the best puck detection
        if puck_conf > best_puck_confidence:
            best_puck_confidence = puck_conf
            best_puck = bbox

        # Check if this is the best stick detection
        if stick_conf > best_stick_confidence:
            best_stick_confidence = stick_conf
            best_stick = bbox

    # Function to draw bounding boxes on frame
    def draw_bbox(frame, bbox, label, confidence):
        cx, cy, w, h = bbox
        img_height, img_width = frame.shape[:2]
        cx, cy, w, h = cx * img_width, cy * img_height, w * img_width, h * img_height
        x, y = int(cx - w / 2), int(cy - h / 2)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw rectangle in red color
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the best puck detection
    if best_puck is not None:
        draw_bbox(frame, best_puck, 'Puck', best_puck_confidence)

    # Draw the best stick detection
    if best_stick is not None:
        draw_bbox(frame, best_stick, 'Stick', best_stick_confidence)

    # Save the frame with annotations
    out.write(frame)

    # Optionally display the frame
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print(f'Annotated video saved to {output_path}')
