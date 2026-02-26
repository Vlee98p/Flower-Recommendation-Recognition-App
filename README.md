---
title: lower-recommendation-recognition-app
app_file: src/app.py
sdk: gradio
sdk_version: 4.44.1
---

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
conda env create -f environment.yml
conda activate flowerapp
pip install -r requirements.txt
```

3. **Launch the App**

Run following command line to run the Gradio interface
```
python src/app.py
```
## 👩🏻‍💻 Model Details
### Supervised Model (Pre-trained)
The flower recognition model is built on **EfficientNet-B0**, a pre-trained CNN from torchvision. Rather than training from scratch, the pre-trained ImageNet weights are leveraged and only the classifier head is replaced with a custom layer (`Linear(1280→64) → ReLU → Dropout → Linear(64→10)`).

Training was done in three stages of progressive fine-tuning:
1. All backbone layers frozen — only the classifier head trained (~80% validation accuracy)
2. Last block (`features.6`) unfrozen with a small learning rate — accuracy improved to ~91% but showed signs of overfitting
3. Two blocks (`features.5` and `features.6`) unfrozen with differential learning rates — stabilized at ~90% accuracy with no overfitting

The final model was selected based on generalization performance rather than peak accuracy.

### Unsupervised Model (Sentence Embeddings)
The bouquet recommendation system uses **`all-MiniLM-L12-v2`** from Sentence Transformers. Each flower entry in the dataset is represented as a rich text description combining its name, occasion, emotion, colors, and composition. These descriptions are encoded into dense vectors and stored as precomputed embeddings (`class_embs.pt`).

At inference time, the user's natural language query is encoded and compared against all flower embeddings using **cosine similarity**. The top 2 most semantically similar bouquets are returned as recommendations. No labeled training data is required — the model relies entirely on the semantic understanding built into the pre-trained transformer.

## Dataset
`Flower Classification | 10 Classes |` dataset from Kaggle is used.

## Results

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
