
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os, base64, random  # ✅ add this line


# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)

# ======================================================
# HELPER — Convert local image to base64
# ======================================================
def img_b64(filename):
    with open(os.path.join("images", filename), "rb") as f:
        return base64.b64encode(f.read()).decode()

species_images = [("fish.png", "50%", 20)]

# ======================================================
# CSS — Background, Bubbles, Fish
# ======================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  height: 100%;
  margin: 0;
  padding: 0;

  /* === Brighter Tropical Ocean to Pale Green Seabed === */
  background: linear-gradient(
    to bottom,
    #e9fcff 0%,     /* pale sky-blue surface */
    #bdf4f6 25%,    /* aqua mid-water */
    #75d3e2 55%,    /* turquoise */
    #3bbad1 75%,    /* deeper blue */
    #b0eac4 100%    /* pale seafoam green seabed */
  );

  overflow: hidden;
}
[data-testid="stAppViewContainer"], .main, .block-container {
  background-color: transparent !important;
}

/* ======== BUBBLES ======== */
.bubble-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  z-index: 0;
  pointer-events: none;
}
.bubble {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(48,179,211,0.7));
  opacity: 0.6;
  animation: floatUp linear infinite;
}
@keyframes floatUp {
  0% { transform: translateY(100vh) scale(0.5); opacity: 0.8; }
  100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
}

/* ======== FISH ======== */
.species-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  pointer-events: none;
  z-index: 3;
}
.species {
  position: absolute;
  width: 200px;
  height: auto;
  opacity: 0.95;
  filter: drop-shadow(0px 3px 6px rgba(0,0,0,0.3));
  animation: swimAndBob 20s linear infinite;
}
@keyframes swimAndBob {
  0%   { transform: translate(-200px, 0) scaleX(-1); }
  25%  { transform: translate(25vw, -15px) scaleX(-1); }
  50%  { transform: translate(100vw, 0) scaleX(-1); }
  50.1%{ transform: translate(100vw, 0) scaleX(1); }
  75%  { transform: translate(25vw, 15px) scaleX(1); }
  100% { transform: translate(-200px, 0) scaleX(1); }
}

/* ======== SEAGRASS ======== */
#seagrass-svg {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100vw;
  height: 420px;
  z-index: 1;
  pointer-events: none;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* 🧭 Bring the main Streamlit content above the animated layers */
[data-testid="stAppViewContainer"] .main, 
.block-container, 
.stFileUploader, 
button, 
div[data-testid="stMarkdownContainer"] {
    position: relative;
    z-index: 10 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SEAGRASS + KELP BALLS
# ======================================================
num_blades = 100
width = 1200
blades, kelp_balls, keyframes = [], [], []

for i in range(num_blades):
    x = random.randint(0, width)
    height = random.randint(180, 300)
    top_shift1 = random.randint(-15, 15)
    top_shift2 = random.randint(-25, 25)
    color = random.choice(["url(#grass1)", "url(#grass2)", "url(#kelp1)"])
    opacity = random.uniform(0.5, 0.9)
    stroke_width = random.uniform(3, 8)
    duration = random.uniform(3, 7)
    delay = random.uniform(0, 4)
    sway_deg = random.uniform(2.5, 5) * (1 if height > 300 else -1)

    anim = f"swayBlade{i}"
    keyframes.append(f"""
    @keyframes {anim} {{
      0%   {{ transform: rotate(-{sway_deg}deg); }}
      100% {{ transform: rotate({sway_deg}deg); }}
    }}
    """)
    
    blades.append(
        f"<path class='grass' d='M{x},420 "
        f"C{x+8},{420 - height * 0.3} {x + top_shift1},{420 - height * 0.55} {x},{420 - height * 0.75} "
        f"S{x + top_shift2},{420 - height * 0.9} {x},{420 - height}' "
        f"fill='none' stroke='{color}' stroke-width='{stroke_width}' opacity='{opacity}' "
        f"style='transform-origin:{x}px 420px; animation:{anim} {duration}s ease-in-out {delay}s infinite alternate;'/>"
    )

# Higher kelp balls (midwater, slightly above seagrass)
for i in range(30):
    x = random.randint(0, width)
    y = random.randint(120, 300)
    size = random.randint(10, 25)
    color = random.choice(["#5cd57c", "#6de28a", "#79eaa0", "#8ff3b2"])
    kelp_balls.append(
        f"<circle cx='{x}' cy='{y}' r='{size/2}' fill='{color}' opacity='0.85' />"
    )

# ======================================================
# SEAGRASS SVG WITH GRADIENTS
# ======================================================
seagrass_svg = f"""
<style>{''.join(keyframes)}</style>
<svg id="seagrass-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 420" preserveAspectRatio="none">
  <defs>
    <linearGradient id="grass1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#34a853"/><stop offset="50%" stop-color="#58c86b"/><stop offset="100%" stop-color="#7ee88b"/>
    </linearGradient>
    <linearGradient id="grass2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2b8c4b"/><stop offset="50%" stop-color="#4fd874"/><stop offset="100%" stop-color="#7af598"/>
    </linearGradient>
    <linearGradient id="kelp1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2e9459"/><stop offset="50%" stop-color="#4ae180"/><stop offset="100%" stop-color="#8ef7a9"/>
    </linearGradient>
  </defs>
  {''.join(blades)}
  {''.join(kelp_balls)}
</svg>
"""
st.markdown(seagrass_svg, unsafe_allow_html=True)

# ======================================================
# ADD BUBBLES
# ======================================================
bubbles_html = (
    "<div class='bubble-container'>"
    + "\n".join([
        f"<div class='bubble' style='left:{i*3.3}%; width:{10+(i%5)*20}px; height:{10+(i%5)*20}px; animation-duration:{5+(i%6)*4}s; animation-delay:{i*0.7}s;'></div>"
        for i in range(40)
    ])
    + "</div>"
)
st.markdown(bubbles_html, unsafe_allow_html=True)

# ======================================================
# ADD FISH
# ======================================================
fish_html = '<div class="species-container">'
for name, top, dur in species_images:
    b64 = img_b64(name)
    fish_html += f"<img class='species' src='data:image/png;base64,{b64}' style='top:{top}; animation-duration:{dur}s;'>"
fish_html += "</div>"
st.markdown(fish_html, unsafe_allow_html=True)



# ==========================================
# FIX: Make Streamlit main container transparent
# ==========================================


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
        Discover the ocean's mysteries. Upload a photo to instantly identify a benthic species<br>
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
def load_model():
    # Initialize model
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))

    # Load trained weights
    model.load_state_dict(torch.load("benthic_classifier.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = load_model()

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
        ax.bar(CLASSES, probs.numpy() * 100, color="#3b82f6")
        [ax.spines[i].set_visible(False) for i in ax.spines]
        ax.tick_params(length = 0)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Class Probabilities")
        ax.get_yaxis().set_visible(False)
        st.pyplot(fig)
else:
    st.info("⬆️ Upload an image to begin classification.")