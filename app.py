
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os, base64, random 
import torch.nn as nn
import os, base64, random  # ✅ add this line
import time
import streamlit as st
from PIL import Image
import base64
import io
import pandas as pd
import numpy as np


# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)



# ✅ Initialize session state once
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Dynamic page title based on selected tab
page_titles = {
    "Home": "Benthic Species Identifier",
    "Classification": "Classification Model",
    "Detection": "Detection Model"
}

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


# ======== SEAGRASS ========
st.markdown("""
<style>
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

# ======== FIX: Keep seagrass behind everything ========
st.markdown("""
<style>
.bubble-container,
.species-container {
    z-index: 0 !important; /* behind everything */
}

#seagrass-svg {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 420px !important;
    z-index: -10 !important; /* always below everything */
    pointer-events: none !important;
}

[data-testid="stAppViewContainer"] .main {
    position: relative;
    z-index: 10 !important;
}

[data-testid="stFileUploader"],
[data-testid="stImage"],
[data-testid="stVerticalBlock"] {
    position: relative;
    z-index: 15 !important;
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

from streamlit import rerun

page_heading = {
    "Home": "Benthic Species Identifier",
    "Classification": "Classification Model",
    "Detection": "Detection Model"
}[st.session_state.page]

description = ""
if st.session_state.page == "Home":
    description = "Discover the ocean's mysteries. Classify benthic species in underwater images with AI-powered recognition."
elif st.session_state.page == "Classification":
    description = (
        "Use this model to identify marine life in underwater photos. "
        "It classifies images into seven benthic species: crab, eel, whelk, scallop, flatfish, roundfish, and skate."
    )
elif st.session_state.page == "Detection":
    description = (
        "Use this model to locate and label marine species in underwater images by "
        "drawing bounding boxes around detected organisms and identifying their species."
    )

st.markdown(f"""
    <style>
    .gradient-text {{
        font-size: 70px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #0D47A1, #1565C0, #64B5F6);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-bottom: 10px;
    }}
    </style>

    <div style='text-align:center;'>
        <span class='gradient-text'>{page_heading}</span>
        <h5 style='text-align:center; color:#37474F;'>
            {description}
        </h5>
    </div>
""", unsafe_allow_html=True)





# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown('<div class="tab-bar">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
        rerun()
with col2:
    if st.button("Classification", use_container_width=True):
        st.session_state.page = "Classification"
        rerun()
with col3:
    if st.button("Detection", use_container_width=True):
        st.session_state.page = "Detection"
        rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap" rel="stylesheet">
<style>
/* Target the actual text inside the Streamlit button */
div[data-testid="stButton"] > button p {
    font-family: 'Poppins', sans-serif !important;
    font-size: 26px !important;      /* ✅ bigger font */
    font-weight: 500 !important;     /* ✅ extra bold */
    color: #1565C0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 50px !important;
            
    
}
</style>
""", unsafe_allow_html=True)


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
CLASSES = ['Eel', 'Scallop', 'Crab', 'Flatfish', 'Roundfish', 'Skate', 'Whelk']
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load("models/benthic_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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




# ======================================================
# PAGE ROUTING
# ======================================================


if st.session_state.page == "Home":
    st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 5rem;
        ">
            <div style="
                background: rgba(255, 255, 255, 0.6);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 18px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 2.5rem 3rem;
                max-width: 800px;
                text-align: center;
                color: #04365c;
            ">
                <h1 style="font-size: 2.6rem; font-weight: 800; color: #0D47A1; margin-bottom: 1rem;">
                    Welcome to AMplify AI 🌊
                </h1>
                <p style="font-size: 1.2rem; line-height: 1.7;">
                    We offer <b>2</b> finetuned deep-learning models for marine life classification and detection.<br><br>
                    Choose our <b>Classification</b> model to identify benthic species in your photos, or our <b>Detection</b> model
                    to box and classify the benthic species in underwater scenes.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)


elif st.session_state.page == "Classification":

    # MAIN CONTENT
    st.markdown("""
    <div style='text-align:center; margin-top:2rem;'>
        <i class="fa-solid fa-upload" style="font-size:55px; color:#1565C0;"></i>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    import time

    # --- Set global layering once, before the upload logic ---
    st.markdown("""
    <style>
    /* Layer hierarchy for ocean visuals */
    .bubble-container,
    #seagrass-svg,
    .species-container {
        z-index: 0 !important; /* very back */
    }

    /* Upload and result boxes - middle layer */
    [data-testid="stFileUploader"],
    .results-box,
    .results-box-graph {
        position: relative !important;
        z-index: 20 !important; /* above kelp/fish */
    }

    /* Loading overlay - topmost layer */
    #loading-overlay {
        z-index: 9999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # 📸 MAIN UPLOAD + LOADING + RESULTS SECTION
    # =========================================================
    import time

    if uploaded_files:
        # ====================================================
        # 🌊 STARFISH LOADING OVERLAY — ALWAYS ON TOP
        # ====================================================
        starfish_b64 = base64.b64encode(open("images/starfish.png", "rb").read()).decode()

        # 1️⃣ Define CSS & JS so overlay beats Streamlit’s container stack
        st.markdown("""
        <style>
        /* BACKGROUND LAYERS */
        .bubble-container, #seagrass-svg, .species-container {
            z-index: 0 !important;
        }

        /* MIDDLE LAYERS (Upload + Results) */
        [data-testid="stFileUploader"], .results-box, .results-box-graph {
            position: relative !important;
            z-index: 10 !important;
        }

        /* TOP LAYER (Loading Screen) */
        #loading-overlay {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(21, 101, 192, 0.55) !important; /* translucent ocean blue */
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 2147483647 !important; /* 🧱 MAXIMUM possible z-index */
            pointer-events: all !important;
        }

        @keyframes spinStar {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)

        # 2️⃣ Overlay HTML — Injected last (so it’s final in DOM)
        loading_html = f"""
        <div id="loading-overlay">
            <img src="data:image/png;base64,{starfish_b64}"
                style="width:120px; height:auto; animation:spinStar 2.5s linear infinite;
                        filter: drop-shadow(0 0 10px rgba(255,255,255,0.8));"/>
            <p style="font-size:1.8rem; font-weight:700; margin-top:1rem; color:white;">Analyzing Marine Life...</p>
            <div style="margin-top:1.5rem; width:300px; height:12px; background:rgba(255,255,255,0.3);
                        border-radius:10px; overflow:hidden;">
                <div id="load-bar" style="width:0%; height:100%;
                            background:linear-gradient(to right, #64B5F6, #1565C0);
                            border-radius:10px;"></div>
            </div>
        </div>
        """

        from components.loading_overlay import show_loading_overlay
        show_loading_overlay("Analyzing Marine Life...", duration=1.0)


        results_all = []  # store results for all images

        from components.species_info import species_info



        total_files = len(uploaded_files)
        for i, uploaded_img in enumerate(uploaded_files):
            # Read image
            img_bytes = uploaded_img.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            species, confidence, probs = predict(image)
            confidence_percent = confidence * 100

            # Store results
            results_all.append({
                "filename": uploaded_img.name,
                "species": species,
                "confidence": confidence_percent,
                "probs": probs
            })

            from components.result_box import render_results_box

# ... inside your loop, where you currently render the HTML ...
            if i < 1:
                from components.species_info import get_species_info

                info = get_species_info(species)

                render_results_box(
                    image_bytes=img_bytes,
                    species=species,
                    confidence_percent=confidence_percent,
                    species_info=info,   # <- a single dict for that species (RIGHT)
                )
    





        
        st.success(f"Processed {total_files} images!")
        # Convert results to DataFrame-friendly format
        for r in results_all:
            # Convert tensor to list of floats
            r["probs"] = [float(p) for p in r["probs"].tolist()]

        
        # Create DataFrame with columns for each class probability
        df_results = pd.DataFrame([
            {
                "Filename": r["filename"],
                "Predicted Species": r["species"],
                "Confidence (%)": round(r["confidence"], 2),
                **{f"P({cls})": round(prob * 100, 2) for cls, prob in zip(CLASSES, r["probs"])}
            }
            for r in results_all
        ])

        st.write("### Classification Results")
        st.dataframe(df_results, use_container_width=True)

        # Convert DataFrame to downloadable CSV
        csv = df_results.to_csv(index=False).encode('utf-8')

        # Download button
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )
            
        st.write("Results for all images are stored in `Classifications` for later use or download.")
        st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        

        from components.auto_scroll import auto_scroll_to_results
        auto_scroll_to_results()



    st.markdown("""
    <style>
    /* === RESET STREAMLIT UPLOADER STYLING === */
    [data-testid="stFileUploader"] section {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        gap: 20px !important;
    }

    [data-testid="stFileUploader"] {
        position: relative !important;
        z-index: 50 !important;
        border-radius: 15px !important;
        padding: 2rem 1rem !important;
        background: rgba(255,255,255,0.45) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* Hide default gray text */
    [data-testid="stFileUploaderLabel"],
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    /* === ADD TITLE ABOVE BUTTON === */
    [data-testid="stFileUploader"]::before {
        content: "Upload Marine Life Photo";
        font-size: 1.6rem;
        font-weight: 700;
        color: #0D47A1;
        display: block;
        margin-bottom: 1rem;
    }

    /* === REBUILD BROWSE BUTTON (REAL BUTTON STYLE) === */
    [data-testid="stFileUploader"] section div div div button {
        appearance: none !important;
        display: inline-block !important;
        border: none !important;
        outline: none !important;
        background: linear-gradient(to right, #1565C0, #1E88E5) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 14px 32px !important;F
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
        text-transform: none !important;
    }

    /* Hover glow */
    [data-testid="stFileUploader"] section div div div button:hover {
        background: linear-gradient(to right, #0D47A1, #1565C0) !important;
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
    }

    /* Ensure button text stays visible */
    [data-testid="stFileUploader"] section div div div button * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* === Fix: Force dataframe above seagrass/fish/bubbles === */
    [data-testid="stDataFrame"] iframe {
        position: relative !important;
        z-index: 9999 !important;
    }

    /* Backup: raise whole dataframe container */
    [data-testid="stDataFrame"] {
        position: relative !important;
        z-index: 9999 !important;
    }

    /* Lower background visuals further if needed */
    #seagrass-svg,
    .species-container,
    .bubble-container {
        z-index: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Detection":

    @st.cache_resource
    def load_detection_model():
        from ultralytics import YOLO
        return YOLO("models/detection_model.pt")  # path to your model

    detection_model = load_detection_model()
    st.info("Upload an image")
    uploaded_file = st.file_uploader(
        "Upload an underwater image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_file:
        from components.loading_overlay import show_loading_overlay
        from components.result_box import render_results_box
        import base64
        from io import BytesIO
        from components.species_info import species_info
        from components.species_info import get_species_info

        show_loading_overlay("Running Detection Model...", duration=1.0)

        results_data = []

        for i, uploaded_img in enumerate(uploaded_file):
            # Read and preprocess image
            image = Image.open(uploaded_img).convert("RGB")
            img_np = np.array(image)

            # Run YOLO detection
            results = detection_model.predict(img_np)
            result = results[0]

            # Draw bounding boxes (NumPy image)
            results_img = result.plot()

            # === Convert annotated image (with boxes) to base64 ===
            results_pil = Image.fromarray(results_img)
            buffer = BytesIO()
            results_pil.save(buffer, format="PNG")
            results_b64 = base64.b64encode(buffer.getvalue()).decode()

            # === Extract top detection info ===
            def get_top_detection(result, names_map):
                """Return (label, confidence%) for top detection."""
                if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
                    return None, None
                confs = result.boxes.conf.cpu().numpy()
                idx = confs.argmax()
                top_conf = float(confs[idx])
                cls_id = int(result.boxes.cls[idx])
                label = names_map.get(cls_id, f"class_{cls_id}")
                return label, top_conf * 100.0
            
            detected_species, conf_percent = get_top_detection(result, detection_model.names)

            # ✅ define info first
            info = get_species_info(detected_species)


            # === Show results box for first image ===
            if i == 0:
                render_results_box(
                    image_bytes=base64.b64decode(results_b64),
                    species=detected_species or "No objects detected",
                    confidence_percent=conf_percent or 0.0,
                    species_info=info
                )


            from components.auto_scroll import auto_scroll_to_results
            auto_scroll_to_results()


            # === Extract all detections for the table ===
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                results_data.append({
                    "Filename": uploaded_img.name,
                    "Class": detection_model.names[cls_id],
                    "Confidence (%)": round(conf * 100, 2),
                    "Bounding Box": [round(x, 2) for x in xyxy]
                })

        # === Create results table ===
        df_results = pd.DataFrame(results_data)
        st.write("### 🐚 Detection Results")
        st.dataframe(df_results, use_container_width=True)

        # === Summary by class ===
        summary_df = (
            df_results.groupby("Class")
            .size()
            .reset_index(name="Total Detections")
            .sort_values(by="Total Detections", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(summary_df, use_container_width=True)

        # === CSV Download ===
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="detection_results.csv",
            mime="text/csv"
        )
