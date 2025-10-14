import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)


with open("app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



st.markdown(
    """
    <h1 style='
        text-align:center;
        font-size:70px; /* 👈 adjust this to make it bigger or smaller */
        font-weight:800;
        margin-bottom:10px;
        background: linear-gradient(to right, #0D47A1, #1565C0, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    '>
        Marine Life Identifier
    </h1>
    <h5 style='text-align:center; color:#37474F;'>
        Discover the ocean's mysteries. Upload a photo to instantly identify marine species<br>
        with AI-powered recognition.
    </h5>
    """,
    unsafe_allow_html=True
)


# CUSTOM STYLES FOR BAR
st.markdown(
    """
    <style>
    .tab-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: white;
        border-radius: 12px;
        overflow: hidden;
        width: 50%;
        margin: 20px auto;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .tab {
        flex: 1;
        text-align: center;
        padding: 12px 0;
        font-size: 18px;
        font-weight: 600;
        color: #1565C0;
        cursor: pointer;
        transition: all 0.3s ease;
        border-right: 1px solid #e0e0e0;
    }
    .tab:last-child {
        border-right: none;
    }
    .tab:hover {
        background-color: #f0f6ff;
        color: #0D47A1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# WHITE BAR WITH TWO SECTIONS
st.markdown(
    """
    <div class="tab-bar">
        <div class="tab">Classification</div>
        <div class="tab">Detection</div>
    </div>
    """,
    unsafe_allow_html=True
)

# SIDEBAR
with st.sidebar:
    st.title("🌊 Benthic AI")
    st.write("""
    This app uses a fine-tuned deep learning model to classify benthic organisms
    (like crabs, scallops, and eels) from underwater imagery.
    """)
    st.markdown("---")
    show_chart = st.checkbox("Show probability chart", value=True)
    st.write("**Created by:** Ariana Robinson & Megan Timmes")
    st.caption("William & Mary AI Case Competition — Fall 2025")


# MODEL SETUP
CLASSES = ["Scallop", "Roundfish", "Crab", "Whelk", "Skate", "Flatfish", "Eel"]

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("benthic_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return CLASSES[idx.item()], conf.item(), probs

# MAIN CONTENT
st.markdown("""
<div style='text-align:center; margin-top:2rem;'>
    <i class="fa-solid fa-upload" style="font-size:55px; color:#1565C0;"></i>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    species, confidence, probs = predict(image)

    st.success("✅ Analysis complete!")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image(image, caption=f"Predicted: {species} ({confidence*100:.1f}%)", use_container_width=True)
    with col2:
        # Display prediction results
        st.metric("Predicted Species", species)

        # Show confidence value and progress bar
        confidence_percent = confidence * 100
        st.write(f"**Confidence:** {confidence_percent:.1f}%")
        st.progress(confidence)


    if show_chart:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(CLASSES, probs.numpy() * 100, color="#3b82f6")
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)
else:
    st.info("⬆️ Upload an image to begin classification.")
