import os
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from mtcnn import MTCNN
import dlib
from torchvision import transforms, models
import torch.nn as nn
from werkzeug.utils import secure_filename
from flask_cors import CORS
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define model architectures
class EfficientNetB0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b0')
        
    def forward(self, x):
        return self.base_model(x)

class ConvNeXtModel(nn.Module):
    def __init__(self):
        super(ConvNeXtModel, self).__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.convnext.classifier = nn.Identity()
        self.feature_processor = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convnext.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.feature_processor(x)

class XceptionModel(nn.Module):
    def __init__(self):
        super(XceptionModel, self).__init__()
        self.base_model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        self.base_model.last_linear = nn.Sequential(
            nn.Linear(self.base_model.last_linear.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models safely
def load_models_safely():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    # Use os.path.join for cross-platform path handling
    model_files = {
        "EfficientNetB0": os.path.join(BASE_DIR, 'model_files', 'f_effb0.pt'),
        "ConvNeXtTiny": os.path.join(BASE_DIR, 'model_files', 'f_convnext.pt'),
        "Xception": os.path.join(BASE_DIR, 'model_files', 'f_xception.pt')
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                print(f"Loading {model_name} from {filepath}")
                model = torch.load(filepath, map_location=device, weights_only=False)
                model.to(device).eval()
                models[model_name] = model
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"Model file not found at {filepath}")
            # List available files in model_files directory for debugging
            model_dir = os.path.join(BASE_DIR, 'model_files')
            if os.path.exists(model_dir):
                print(f"Available files in {model_dir}:")
                for file in os.listdir(model_dir):
                    print(f"  - {file}")
            else:
                print(f"Model directory {model_dir} does not exist")
    
    return models, device

# Load models once
models, device = load_models_safely()

# Load dlib's face detector and shape predictor safely
try:
    dlib_detector = dlib.get_frontal_face_detector()
    shape_predictor_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks_GTX.dat')
    if os.path.exists(shape_predictor_path):
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        print("Dlib shape predictor loaded successfully")
    else:
        print(f"Shape predictor file not found at {shape_predictor_path}")
        shape_predictor = None
except Exception as e:
    print(f"Error loading dlib components: {e}")
    dlib_detector = None
    shape_predictor = None

# Preprocess image
def preprocess_image(image, model_name):
    if model_name == "EfficientNetB0":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == "ConvNeXtTiny":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == "Xception":
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return transform(image).unsqueeze(0)

# Extract face using dlib's shape predictor
def crop_face_with_landmarks(frame, rect):
    if shape_predictor is None:
        # Fallback: return the detected face rectangle
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        return frame[y:y+h, x:x+w]
    
    # Get the landmarks
    landmarks = shape_predictor(frame, rect)
    x_min = min([landmarks.part(i).x for i in range(68)])
    x_max = max([landmarks.part(i).x for i in range(68)])
    y_min = min([landmarks.part(i).y for i in range(68)])
    y_max = max([landmarks.part(i).y for i in range(68)])
    
    # Crop the face based on the landmarks
    return frame[y_min:y_max, x_min:x_max]

# Predict video authenticity for a single model
def predict_video(video_path, model, model_name, num_frames=15):
    mtcnn_detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    predictions = []
    confidences = []  # List to store confidence scores

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {"prediction": "Error: Could not read video", "confidence": None}

    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_mtcnn = mtcnn_detector.detect_faces(rgb_frame)

            if faces_mtcnn:  # Use MTCNN for face detection
                x, y, w, h = faces_mtcnn[0]['box']
                face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w]).convert('RGB')
            elif dlib_detector is not None:  # Fallback to dlib if MTCNN fails
                dlib_faces = dlib_detector(rgb_frame, 1)
                if dlib_faces:
                    d = dlib_faces[0]
                    cropped_face = crop_face_with_landmarks(rgb_frame, d)
                    face_img = Image.fromarray(cropped_face).convert('RGB')
                else:
                    continue  # Skip if no face detected
            else:
                continue  # Skip if no face detection method available
                
            input_tensor = preprocess_image(face_img, model_name).to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(input_tensor)
                confidence = output.item()  # Extract confidence score
                predictions.append((output > 0.5).int().item())
                confidences.append(confidence)

    cap.release()
    if not predictions:
        return {"prediction": "No faces detected", "confidence": None}

    # Compute final prediction and confidence
    final_prediction = "Fake" if sum(predictions) > len(predictions) / 2 else "Real"
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Adjust confidence for interpretation
    if final_prediction == "Fake":
        adjusted_confidence = avg_confidence  # Scale to percentage for "Fake" predictions
    else:
        adjusted_confidence = (1 - avg_confidence)   # Invert confidence for "Real" predictions

    return {
        "prediction": final_prediction,
        "confidence": round(adjusted_confidence, 2)  # Round to 2 decimal places
    }

# Define routes
@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({"error": "No models loaded"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        predictions = {}
        for model_name, model in models.items():
            result = predict_video(filepath, model, model_name)
            predictions[model_name] = {
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            }

        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/')
def home():
    return jsonify({
        "message": "Video Authenticity Detector Backend is Running",
        "models_loaded": list(models.keys()),
        "total_models": len(models)
    }), 200

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models_loaded": len(models)}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)