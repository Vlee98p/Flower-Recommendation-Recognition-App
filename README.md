
# 🌸 Welcome to Flower Recommendation & Recognition App

## 🌷 About
The **Flower Recommendation & Recognition App** is a small, end‑to‑end machine learning project designed to make flower discovery simple and delightful. It combines natural language understanding with computer vision to help users either:

- Find the right bouquet based on a mood, message, or occasion
- Identify a flower from an uploaded image

The text recommendation system uses sentence embeddings to interpret user intent and match it with curated flower descriptions. The image classifier is a lightweight CNN trained on a custom dataset of flower species.

This project was built to demonstrate practical ML deployment, clean UI design, and the ability to ship a polished, user‑facing tool.

## 🪻 Features
- Sentence‑embedding–based bouquet recommendations
- CNN‑based flower image classification
- Clean Gradio interface with custom styling
- Lightweight, fast, and beginner‑friendly

## 🌺 Built With
- Python
- PyTorch (CNN model)
- Sentence Transformers (text embeddings)
- Gradio (UI)
- YAML / JSON (config + metadata)

## 🌹 How to Run
1. **Clone this Repository**
```
git clone git@github.com:Vlee98p/Flower-Recommendation-Recognition-App.git
```

2. **Install dependencies**
Navigate to the root of this project and run the following command line
```
pip install -r requirements.txt
```

3. **Launch the App**
Run following command line to run the Gradio interface
```
python app.py
```

## Developer Notes
- `Text Embeddings`: Generated using Sentence Transformers to map user input to flower descriptions.
- `Image Classifier`: A lightweight CNN trained on a curated flower dataset.

### Metadata Files:
- `flower_labels.json` maps class indices to flower names
- `flower_colour.json` stores color associations
- `flowers.yaml` defines bouquet rules and mappings

### Models:

- `class_embs.pt` stores precomputed text embeddings
- `cnn_flower_classifier.pth` is the trained CNN model

### UI
- Built with Gradio Blocks and custom CSS, including an animated sunflower mascot positioned above the tab bar.

## Dependencies
- Python 3.10+
- PyTorch
- Sentence Transformers
- Gradio
- PyYAML
- JSON
