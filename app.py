import gradio as gr
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load the trained model
def load_model():
    num_classes = 2
    model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("model/karuvelam_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define the image transformations
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference function
def predict(image):
    # Process the image
    image_tensor = transform(image).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Process the output to draw bounding boxes
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    # Draw boxes on the image
    image_np = np.array(image)
    for box, score in zip(boxes, scores):
        if score > 0.5: # Set a confidence threshold
            xmin, ymin, xmax, ymax = box.astype(int)
            image_np = cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_np, f'Score: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_np

# Create the Gradio interface
iface = gr.Interface(fn=predict, inputs="image", outputs="image", title="Karuvelam Tree Detector")
iface.launch()
