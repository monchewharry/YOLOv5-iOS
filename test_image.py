import coremltools as ct
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the Core ML model
model = ct.models.MLModel('./runs/train/exp/weights/best.mlpackage')

# Load and preprocess the image
image_path = '../datasets/hockeypuck/images/test/height120_frame_0001.png'
image = Image.open(image_path).convert('RGB')  # Convert to RGB
image = image.resize((640, 640))  # Resize the image

# Create input dictionary using the PIL image directly
input_dict = {'image': image}

# Perform inference
result = model.predict(input_dict)
print(result)
"""
{'coordinates': array([[0.37499994, 0.82739896, 0.05103123, 0.07589915],
       [0.830599  , 0.5616406 , 0.04652033, 0.04080057],
       [0.32679477, 0.77860206, 0.05272928, 0.12222137]], dtype=float32), 
'confidence': array([[0.8585572 , 0.09743853],
       [0.5303041 , 0.02289737],
       [0.03158174, 0.4365411 ]], dtype=float32)}
"""

# Extract information from the result
coordinates = result['coordinates']
confidence = result['confidence']

# Original image size
img_width, img_height = image.size

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

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(image)
ax = plt.gca()

# Function to draw bounding boxes
def draw_bbox(bbox, label, confidence):
    cx, cy, w, h = bbox
    cx, cy, w, h = cx * img_width, cy * img_height, w * img_width, h * img_height
    x, y = cx - w / 2, cy - h / 2
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(x, y, f'{label} {confidence:.2f}', color='white', fontsize=12, backgroundcolor='red')

# Draw the best puck detection
if best_puck is not None:
    draw_bbox(best_puck, 'Puck', best_puck_confidence)

# Draw the best stick detection
if best_stick is not None:
    draw_bbox(best_stick, 'Stick', best_stick_confidence)

plt.axis('off')
plt.show()
