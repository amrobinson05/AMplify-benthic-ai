import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# --- CONFIG ---
st.set_page_config(page_title="Benthic AI", page_icon="🌊", layout="centered")

# --- CLASSES ---
CLASSES = ["Scallop", "Roundfish", "Crab", "Whelk", "Skate", "Flatfish", "Eel"]

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("benthic_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- PREDICT FUNCTION ---
def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return CLASSES[idx.item()], conf.item()

# --- HEADER ---
st.title("🌊 Benthic AI: Marine Species Identification")
st.write("""
Upload an underwater image, and our fine-tuned AI model will classify it into one of seven benthic species:
**Scallop, Roundfish, Crab, Whelk, Skate, Flatfish, Eel**.
""")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.info("⏳ Analyzing image...")
    species, confidence = predict(image)

    st.success("✅ Analysis complete!")
    st.markdown(f"### **Predicted Species:** {species}")
    st.markdown(f"**Confidence:** {confidence * 100:.1f}%")

else:
    st.info("⬆️ Upload an image to classify it.")

# --- FOOTER ---
st.write("---")
st.caption("Built by Ariana Robinson & Megan Timmes — William & Mary AI Case Competition 2025 🌊")
