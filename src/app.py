import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
import json
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import traceback
from PIL import Image
import os

# Load your models
#sentence embedding
with open("config/flower_labels.json") as f:
    flower_classes = json.load(f)

with open("config/flower_colour.json") as f:
    flower_colour = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
class_embs = embedding_model.encode(flower_classes, convert_to_tensor=True)


def recommend_bouquet(text):
    try:
        if not text.strip():
            return "Please describe your thoughts or occasion."
        query_emb = embedding_model.encode(text, convert_to_tensor=True)

        scores = torch.nn.functional.cosine_similarity(query_emb, class_embs)
        print("scores shape:", scores.shape)

        top2_idx = torch.topk(scores, k=2).indices.tolist()
        print("top2_idx:", top2_idx)

        best_idx = top2_idx[0]
        second_idx = top2_idx[1]

        return (
            f"🌸 First option: {flower_classes[best_idx]}, colour: {flower_colour[best_idx]}\n\n"
            f"🌸 Second option: {flower_classes[second_idx]}, colour: {flower_colour[second_idx]}")

    except Exception as e:
        traceback.print_exc()
        return f"Internal error: {e}"

#CNN
def load_cnn_model():
    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(1280, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 10)
    )

    # Load your trained weights
    state_dict = torch.load("models/cnn_flower_classifier.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

cnn_model = load_cnn_model()   # EfficientNet architecture + trained weights from Kaggle

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_flower(file_obj):
    try:
        if file_obj is None:
            return "⚠️ **Please upload a flower image.**"

        file_path = file_obj.name

        # Try to open the image manually
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception:
            return "❌ **This image format is not supported. Please upload a JPG or PNG file.**"

        img = transform(image).unsqueeze(0).float()

        with torch.no_grad():
            outputs = cnn_model(img)
            predicted_idx = outputs.argmax(dim=1).item()

        predicted_label = flower_classes[predicted_idx]

        return f"🌸 {predicted_label}"

    except Exception as e:
        traceback.print_exc()
        return f"❌ **Internal error:** {e}"


# Add this near the top of your app.py
gif_path_s = "/app/assets/Sunflowers.gif"
gif_path_t = "/app/assets/Tulip.gif"

with gr.Blocks(css="""
    .gradio-container {
        max-width: 700px !important;
        min-width: 700px !important;
        margin: auto !important; """) as demo:
    gr.Markdown("# 🌸 Flower Recommendation & Recognition App 🌸")

    with gr.Column(scale=4):
    
        gr.HTML(f'''
            <div style="position:relative; height:0;">
                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:250px;">
                <img src="/file={gif_path_t}" width="35" style="position:absolute; top:10px; left:300px;">

                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:310px;">
                <img src="/file={gif_path_t}" width="35" style="position:absolute; top:10px; left:360px;">

                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:370px;">
                <img src="/file={gif_path_t}" width="35" style="position:absolute; top:10px; left:420px;">

                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:430px;">
                <img src="/file={gif_path_t}" width="35" style="position:absolute; top:10px; left:480px;">

                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:490px;">
                <img src="/file={gif_path_t}" width="35" style="position:absolute; top:10px; left:540px;">

                <img src="/file={gif_path_s}" width="80" style="position:absolute; top:-5px; left:550px;">
            </div>
        ''')

        with gr.Tabs():
            with gr.Tab("Text to Bouquet"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Describe your thoughts or occasion",
                            lines=3,
                        )
                        text_output = gr.Markdown("### Recommendation")
                        text_button = gr.Button("Recommend", variant="primary")
                        text_button.click(
                            fn=recommend_bouquet,
                            inputs=text_input,
                            outputs=text_output
                        )

            with gr.Tab("Image to Flower"):
                #gr.HTML(f'<div style="text-align:center;"><img src="/file={gif_path}" width="100"></div>')
                img_file = gr.File(file_types=["image"])
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Predicted flower")
                        img_output = gr.Markdown()
                        img_button = gr.Button("Identify Flower", variant="primary")
                        img_button.click(
                            fn=classify_flower,
                            inputs=img_file,
                            outputs=img_output
                        )
  
demo.launch(allowed_paths=["/app/assets"])
