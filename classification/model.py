import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# --- Define your model's label classes ---
CLASSES = ["Scallop", "Roundfish", "Crab", "Whelk", "Skate", "Flatfish", "Eel"]

# --- Cache model loading so it only happens once per session ---
@st.cache_resource
def load_classification_model():
    """
    Load the fine-tuned ResNet model used for species classification.
    Cached to avoid reloading on every rerun.
    """
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("models/classification_model.pth", map_location="cpu"))
    model.eval()
    return model


# --- Define preprocessing transforms ---
def get_transforms():
    """Return the preprocessing transform pipeline used for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# --- Prediction function ---
def predict(image):
    """
    Run the classification model on an input PIL image.

    Returns:
        (species_name, confidence_value, probability_tensor)
    """
    model = load_classification_model()
    transform = get_transforms()

    # Preprocess image
    tensor = transform(image).unsqueeze(0)

    # Run model inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, cls = probs.max(1)

    species = CLASSES[cls.item()]
    confidence = conf.item()

    return species, confidence, probs.squeeze()
