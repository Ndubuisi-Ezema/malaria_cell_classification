# Malaria Cell Image Research Studio

Interactive Streamlit application based on the accompanying malaria cell image classification notebook.

## Features
- Single-image prediction with adjustable Parasitized threshold
- Batch prediction with downloadable CSV
- Dataset explorer with class balance, sample gallery and augmentation preview
- Labelled-dataset evaluation: accuracy, precision, recall, F1, confusion matrix, classification report, ROC and precision-recall curves
- Transfer-learning training: EfficientNetB0, VGG16 or ResNet50
- Grad-CAM heatmap and overlay for supported models
- Download trained Keras model

## Run locally
Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run ctg_app.py
```

The app entrypoint in this package is `app.py`, so run:

```bash
python -m streamlit run app.py
```

## Dataset ZIP format
```text
dataset.zip
└── cell_images/
    ├── Parasitized/
    └── Uninfected/
```
The two class folders can also be at the ZIP root. Supported image formats: JPG, JPEG, PNG, BMP, WEBP.

## Model upload
Upload a TensorFlow/Keras `.keras` or `.h5` file in the sidebar. The model should accept 224x224 RGB images scaled to [0,1] and return a single sigmoid output. The notebook's default Keras alphabetical folder order is 0 = Parasitized, 1 = Uninfected. The notebook's classification-report target-name order conflicts with this, so verify the mapping before interpreting results.

## Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app using the repository, branch `main`, and entrypoint `app.py`.
3. Select Python 3.11 if offered.
4. Deploy. TensorFlow installation can be large and may exceed some free hosting resource limits; local deployment is an alternative.

## Safety
Research/education only. Not a medical device and not validated for clinical use. Never use predictions for diagnosis or treatment decisions. Do not upload patient-identifiable data unless your institution has approved the data handling and hosting arrangements.
