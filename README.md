# AI_DD: Deepfake Face Detection

AI_DD is a deepfake face detection project that leverages state-of-the-art deep learning models to identify manipulated facial videos and images. The repository integrates three powerful pre-trained models—ConvNeXt Tiny, EfficientNet-B0, and Xception—trained on large-scale datasets from Kaggle to ensure robust detection capabilities.

## Features

- **Deepfake Detection:** Utilizes three advanced pre-trained models:
  - ConvNeXt Tiny
  - EfficientNet-B0
  - Xception
- **Dataset:** Models are trained and fine-tuned using Kaggle deepfake datasets for improved accuracy and generalization.
- **Full Stack Application:**
  - **Frontend:** Built with React for an interactive and responsive user experience.
  - **Backend:** Developed using Python to handle model inference and API endpoints.
- **Modern Web Technologies:** JavaScript is used extensively for frontend logic, with CSS and HTML for styling and layout.

## Getting Started

### Prerequisites

- Node.js and npm (for the frontend)
- Python 3.x (for the backend)
- (Optional) Kaggle API credentials if you wish to retrain or experiment with datasets

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dp1345/AI_DD.git
   ```

2. **Frontend Setup (React):**
   ```bash
   cd AI_DD/frontend
   npm install
   npm start
   ```

3. **Backend Setup (Python):**
   ```bash
   cd AI_DD/backend
   pip install -r requirements.txt
   python app.py
   ```

4. **Usage:**
   - Access the frontend app in your browser (usually at `http://localhost:3000`)
   - Upload or stream a video/image for deepfake detection

## Project Structure

- `frontend/` – React app for user interface
- `backend/` – Python backend for model inference and API logic
- `models/` – Pre-trained model weights and architecture definitions (not included in repo)
- `src/` – Core logic (mainly JavaScript)
- `styles/` – CSS files for design
- `index.html` – Main entry point for the web UI

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.



*Created by [dp1345](https://github.com/dp1345)*
