
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
from components.backgrounds.ocean_home import render_ocean_home
from components.backgrounds.deep_detection import render_deep_sea
from collections import Counter
import math
import seaborn as sns

# 🌿 Ecological Role Dictionary for All 7 Benthic Species
KEYSTONE_SPECIES = {
    "Scallop": "", "Roundfish": "", "Crab": "",
    "Whelk": "", "Skate": "", "Flatfish": "", "Eel": ""
}



from utils.ecosystem import (
    calculate_ecosystem_health,
    plot_biodiversity_pie,
    plot_species_abundance
)



# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
/* 🚫 Disable Streamlit fade / slide animations */
[data-testid="stAppViewContainer"], 
[data-testid="stVerticalBlock"], 
[data-testid="stHorizontalBlock"], 
[data-testid="stMarkdownContainer"],
[data-testid="stColumn"] {
    transition: none !important;
    animation: none !important;
}

/* Also disable animations for buttons and reruns */
button, [role="button"], [data-testid="stButton"] {
    transition: none !important;
}

/* Avoid fade effects on images */
img:not(.species):not(.spinStar):not(.jellyfish) {
    transition: none !important;
    animation: none !important;
}

/* Disable Streamlit spinner fade too */
[data-testid="stSpinner"] {
    animation: none !important;
}
</style>
""", unsafe_allow_html=True)


# ✅ Initialize session state once
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Add a title dictionary (optional)
page_titles = {
    "Home": "Benthic Species Identifier",
    "Classification": "Classification Model",
    "Detection": "Detection Model",
    "Metrics": "Model Metrics"   # ✅ NEW PAGE
}


# ==========================================
# FIX: Make Streamlit main container transparent
# ==========================================


from streamlit import rerun

page_heading = {
    "Home": "Benthic Species Identifier",
    "Classification": "Classification Model",
    "Detection": "Detection Model",
    "Metrics": "Model Metrics"   # ✅ added
}[st.session_state.page]

description = ""
if st.session_state.page == "Home":
    render_ocean_home()
    with open("app.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    description = "Discover the ocean's mysteries. Classify benthic species in underwater images with AI-powered recognition."
elif st.session_state.page == "Classification":
    render_deep_sea()
    st.markdown("""
<style>
/* === Deep Blue File Uploader Box === */
[data-testid="stFileUploader"] {
    background: rgba(0, 24, 61, 0.6) !important;  /* translucent navy */
    border: 3px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 25px rgba(0, 40, 80, 0.5), inset 0 0 10px rgba(0, 80, 160, 0.3) !important;
    width: 60% !important;
    height: 400px !important;
    margin: 2rem auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* === File Uploader Label/Subheading === */
[data-testid="stFileUploader"]::before {
    content: "Upload Marine Life Photo";
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff !important; /* White subheading */
    margin-bottom: 30px;
    display: block;
}

/* === Browse Files Button (Darker with Glow) === */
[data-testid="stFileUploader"] section div div div button {
    background: linear-gradient(to right, #00142a, #002b5b) !important; /* darker navy */
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 36px !important;
    box-shadow: 0 0 18px rgba(0, 100, 200, 0.25) !important;
    transition: all 0.3s ease-in-out !important;
}
[data-testid="stFileUploader"] section div div div button:hover {
    background: linear-gradient(to right, #003566, #004b8d) !important;
    box-shadow: 0 0 30px rgba(0, 150, 255, 0.6) !important;
    transform: scale(1.05);
}
.stButton > button {
    background: #001d3d !important; /* solid dark navy */
    color: #ffffff !important;      /* white text */
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 0.6rem 2.2rem !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 0 12px rgba(0, 50, 100, 0.3) !important;
}}
.stButton > button:hover {
    background: linear-gradient(to right, #003566, #004b8d) !important;
    box-shadow: 0 0 25px rgba(0, 150, 255, 0.6) !important;
    transform: scale(1.03);
}

/* === Text Fix (White in Dark Theme) === */
h1, h2, h3, h4, h5, label, p, span, div {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)
    
    description = (
        "Use this model to identify marine life in underwater photos. "
        "It classifies images into seven benthic species: crab, eel, whelk, scallop, flatfish, roundfish, and skate."
        
    )
elif st.session_state.page == "Detection":
    render_deep_sea()
    st.markdown("""
<style>
/* === Deep Blue File Uploader Box === */
[data-testid="stFileUploader"] {
    background: rgba(0, 24, 61, 0.6) !important;  /* translucent navy */
    border: 3px dashed rgba(255, 255, 255, 0.5) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 25px rgba(0, 40, 80, 0.5), inset 0 0 10px rgba(0, 80, 160, 0.3) !important;
    width: 60% !important;
    height: 400px !important;
    margin: 2rem auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* === File Uploader Label/Subheading === */
[data-testid="stFileUploader"]::before {
    content: "Upload Marine Life Photo";
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff !important; /* White subheading */
    margin-bottom: 30px;
    display: block;
}

/* === Browse Files Button (Darker with Glow) === */
[data-testid="stFileUploader"] section div div div button {
    background: linear-gradient(to right, #00142a, #002b5b) !important; /* darker navy */
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 36px !important;
    box-shadow: 0 0 18px rgba(0, 100, 200, 0.25) !important;
    transition: all 0.3s ease-in-out !important;
}
[data-testid="stFileUploader"] section div div div button:hover {
    background: linear-gradient(to right, #003566, #004b8d) !important;
    box-shadow: 0 0 30px rgba(0, 150, 255, 0.6) !important;
    transform: scale(1.05);
}
.stButton > button {
    background: #001d3d !important; /* solid dark navy */
    color: #ffffff !important;      /* white text */
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 0.6rem 2.2rem !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 0 12px rgba(0, 50, 100, 0.3) !important;
}}
.stButton > button:hover {
    background: linear-gradient(to right, #003566, #004b8d) !important;
    box-shadow: 0 0 25px rgba(0, 150, 255, 0.6) !important;
    transform: scale(1.03);
}

/* === Text Fix (White in Dark Theme) === */
h1, h2, h3, h4, h5, label, p, span, div {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

    description = (
        "Use this model to locate and label marine species in underwater images by "
        "drawing bounding boxes around detected organisms and identifying their species."
    )

elif st.session_state.page == "Metrics":
    with open("app.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    render_ocean_home()
    description = ("Explore model accuracy, precision, and performance results."
    )

# 🌊 Title Styling — gradient for normal pages, glow for Classification/Detection
st.markdown(f"""
    <style>
    /* === Base Gradient Text (used for Home & Metrics) === */
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

    /* === Glowing Text (used for Classification & Detection) === */
    .glow-title {{
        font-size: 70px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #5fa8f3, #1e88e5, #90caf9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow:
            0 0 4px rgba(100,181,246,0.6),
            0 0 10px rgba(33,150,243,0.3),
            0 0 20px rgba(33,150,243,0.2);
        animation: gentlePulse 5s ease-in-out infinite;
        display: inline-block;
        margin-bottom: 10px;
    }}

    @keyframes glowPulse {{
        0%, 100% {{
            text-shadow:
                0 0 4px rgba(100,181,246,0.6),
                0 0 10px rgba(33,150,243,0.3),
                0 0 20px rgba(33,150,243,0.2);
        }}
        50% {{
            text-shadow:
                0 0 6px rgba(144,202,249,0.8),
                0 0 14px rgba(33,150,243,0.5),
                0 0 24px rgba(33,150,243,0.3);
        }}
    }}
    </style>
""", unsafe_allow_html=True)
# Use glowing title for Classification and Detection pages
glow_class = "glow-title" if st.session_state.page in ["Classification", "Detection"] else "gradient-text"

st.markdown(f"""
    <div style='text-align:center;'>
        <span class='{glow_class}'>{page_heading}</span>
        <h5 style='text-align:center; color:#37474F;'>
            {description}
        </h5>
    </div>
""", unsafe_allow_html=True)


# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "classification_nonce" not in st.session_state:
    st.session_state.classification_nonce = 0
if "detection_nonce" not in st.session_state:
    st.session_state.detection_nonce = 0


# --- Tab bar layout ---
st.markdown('<div class="tab-bar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Home", use_container_width=True):
        st.session_state.classification_nonce += 1
        st.session_state.detection_nonce += 1
        st.session_state.page = "Home"
        st.rerun()
with col2:
    if st.button("Classification", use_container_width=True):
        st.session_state.classification_nonce += 1
        st.session_state.page = "Classification"
        st.rerun()
with col3:
    if st.button("Detection", use_container_width=True):
        st.session_state.detection_nonce += 1
        st.session_state.page = "Detection"
        st.rerun()
with col4:
    if st.button("Metrics", use_container_width=True):
        st.session_state.classification_nonce += 1
        st.session_state.detection_nonce += 1
        st.session_state.page = "Metrics"
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# 🎨 NAVIGATION BUTTON STYLING (White + Shadow)
# ======================================================
st.markdown("""
<style>
/* === STREAMLIT BUTTON STYLE OVERRIDE === */
.stButton > button {
    background-color: white !important;
    color: #1565C0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    width: 100% !important;            /* ✅ uniform width */
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
}

/* Hover effect — light blue tint */
.stButton > button:hover {
    background-color: #f2f7ff !important;
    transform: scale(1.03);
    box-shadow: 0 10px 18px rgba(0,0,0,0.65);
}

/* Active (pressed) look */
.stButton > button:active {
    transform: scale(0.97);
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}

/* Keep buttons centered and even */
.tab-bar {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


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



# MODEL SETUP
CLASSES = ['Eel', 'Scallop', 'Crab', 'Flatfish', 'Roundfish', 'Skate', 'Whelk']
def load_model():
    model = models.efficientnet_v2_s(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load("models/classification_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((384, 384)),  # V2-S input
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




import streamlit.components.v1 as components

if st.session_state.page == "Home":
    st.markdown("""
        <style>
        @keyframes floatCard {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
            100% { transform: translateY(0px); }
        }
        .floating-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            padding: 3rem 3.5rem;
            max-width: 850px;
            margin: 6rem auto;
            text-align: center;
            color: #04365c;
            animation: floatCard 6s ease-in-out infinite;
        }
        </style>

        <div class="floating-card">
            <h1 style="font-size: 2.8rem; font-weight: 900; color: #0D47A1; margin-bottom: 0.6rem;">
                Welcome to <span style="color:#1565C0;">AMplify AI 🌊</span>
            </h1>
            <p style="font-size: 1rem; color: #1565C0; margin-bottom: 1.8rem; font-weight: 500;">
                <b>Created by:</b> <span style="color:#0D47A1;">Ariana Robinson</span> &amp; 
                <span style="color:#0D47A1;">Megan Timmes</span><br>
                <span style="font-size: 0.95rem; color: #04365c;">
                    William &amp; Mary AI Case-a-thon — <b>Fall 2025</b>
                </span>
            </p>
            <p style="font-size: 1.25rem; line-height: 1.8; color: #04365c;">
                We offer <b style="color:#0D47A1;">two fine-tuned deep learning models</b> 
                for marine life <b style="color:#1565C0;">classification</b> and 
                <b style="color:#1565C0;">detection</b>.<br><br>
                Use the <b style="color:#0D47A1;">Classification</b> model to 
                identify benthic species in your photos, or the 
                <b style="color:#0D47A1;">Detection</b> model to 
                locate and label organisms in full underwater scenes.
            </p>
        </div>
    """, unsafe_allow_html=True)




elif st.session_state.page == "Classification":

    # MAIN CONTENT
    st.markdown("""
    <div style='text-align:center; margin-top:2rem;'>
        <i class="fa-solid fa-upload" style="font-size:55px; color:#1565C0;"></i>
    </div>
    """, unsafe_allow_html=True)

    # --- In Classification page ---
    uploaded_files = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"classification_upload_v{st.session_state.classification_nonce}"
)

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

    # === Ocean-themed chart container and text glow ===
    st.markdown("""
    <style>
    /* Chart wrapper with translucent background and glow */
    .canvas-container {
        background: rgba(0, 24, 61, 0.45);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 0 25px rgba(33,150,243,0.25);
        margin-top: 1.5rem;
    }

    /* Section titles (e.g. "Ecological Insights") */
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        color: #E3F2FD !important;
        text-shadow: 0 0 8px rgba(21,101,192,0.7);
    }

    /* Paragraph text inside analysis sections */
    div[data-testid="stMarkdownContainer"] p {
        color: #E3F2FD !important;
    }

    /* Streamlit table tweaks (optional) */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(13,71,161,0.2);
    }
    </style>
    """, unsafe_allow_html=True)



    import time

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]

        # 🧠 Detect if new files were ADDED (not just removed or re-ordered)
        prev_names = st.session_state.get("last_uploaded_classification", [])
        added_files = [f for f in current_names if f not in prev_names]

        if added_files:
            # Update session for classification uploads
            st.session_state.last_uploaded_classification = current_names
            st.session_state.scrolled_after_upload = False

            from components.loading_overlay import show_loading_overlay
            show_loading_overlay("Analyzing Marine Life...", duration=1.0)
        else:
            # Just update the list quietly (no overlay when removing)
            st.session_state.last_uploaded_classification = current_names



        # Continue with starfish overlay CSS (this part stays)
        starfish_b64 = base64.b64encode(open("images/starfish.png", "rb").read()).decode()

        st.markdown("""
        <style>
        /* BACKGROUND LAYERS */
        .bubble-container, #seagrass-svg, .species-container {
            z-index: 0 !important;
        }
        [data-testid="stFileUploader"], .results-box, .results-box-graph {
            position: relative !important;
            z-index: 10 !important;
        }
        #loading-overlay {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(21, 101, 192, 0.55) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 2147483647 !important;
            pointer-events: all !important;
        }
        @keyframes spinStar {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)



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
            if i < 1:
                from components.species_info import get_species_info
                from components.auto_scroll import auto_scroll_to_results

                # Only scroll once per new upload batch
                if not st.session_state.get("scrolled_after_upload", False):
                    st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)
                    auto_scroll_to_results()
                    st.session_state.scrolled_after_upload = True

                info = get_species_info(species)
                render_results_box(
                    image_bytes=img_bytes,
                    species=species,
                    confidence_percent=confidence_percent,
                    species_info=info,
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

        # 🌊 Optional Ecological Insights toggle
        show_insights = st.checkbox("Show Ecological Insights", value=False)

        if show_insights:
            st.markdown("## 🌿 Ecological Insights")

            # --- Core Calculations ---
            diversity_score, status = calculate_ecosystem_health(results_all)
            species_list = [r.get("species") for r in results_all if r.get("species")]
            species_counts = Counter(species_list)
            total = sum(species_counts.values())
            num_species = len(species_counts)

            # --- Keystone Species ---
            keystone_found = [s for s in species_list if s.capitalize() in KEYSTONE_SPECIES]
            keystone_count = len(set([s.capitalize() for s in keystone_found]))

            # --- Evenness (Pielou’s J) ---
            if num_species > 1:
                evenness = (diversity_score * math.log(len(species_counts))) / math.log(num_species)
            else:
                evenness = 0.0

            # --- Dominant Species ---
            if species_counts:
                dominant_species = max(species_counts, key=species_counts.get)
                dominant_percent = (species_counts[dominant_species] / total) * 100
            else:
                dominant_species, dominant_percent = "None", 0

            # --- Biodiversity Tier ---
            if keystone_count >= 6:
                box_color = "#2e7d32"
                biodiversity_text = "🌿 Strong biodiversity — most key species present."
            elif keystone_count >= 3:
                box_color = "#fbc02d"
                biodiversity_text = "🪸 Moderate biodiversity — some key species detected."
            elif keystone_count >= 1:
                box_color = "#ef6c00"
                biodiversity_text = "⚠️ Limited biodiversity — few key species detected."
            else:
                box_color = "#c62828"
                biodiversity_text = "🚫 No key species identified."

            # --- Ecological Summary ---
            if diversity_score > 0.75 and evenness > 0.6:
                summary_text = "Stable ecosystem — balanced and resilient community structure."
            elif diversity_score > 0.4:
                summary_text = "Moderately stable ecosystem with partial species balance."
            else:
                summary_text = "Low stability — one or two species dominate the environment."

            # --- Unified Tier-Colored Display Box ---
            st.markdown(f"""
            <div style='
                background:rgba(0, 24, 61, 0.85);
                border-radius:15px;
                border:2px solid {box_color};
                box-shadow:0 8px 25px rgba(0,0,0,0.4);
                backdrop-filter:blur(12px);
                padding:1.8rem;
                margin-bottom:1.5rem;
                color:#E3F2FD;
                text-align:center;
            '>
                <h3 style='color:#64B5F6;'>Ecosystem Overview</h3>
                <p style='font-size:1.05rem; color:#90CAF9; margin-bottom:0.8rem;'><b>{status}</b></p>
                <hr style='border:0; height:1px; background:rgba(144,202,249,0.4); margin:1rem 0;'>
                <p style='font-size:0.95rem; line-height:1.6; color:#BBDEFB;'>
                    <b>Shannon Index:</b> {diversity_score:.3f} • 
                    <b>Evenness:</b> {evenness:.3f} • 
                    <b>Indicator Species:</b> {keystone_count}/7<br>
                    <b>Dominant Species:</b> {dominant_species} ({dominant_percent:.1f}%)
                </p>
                <p style='font-size:0.95rem; color:#B3E5FC; margin-top:1.2rem; line-height:1.6;'>
                    {biodiversity_text}<br>{summary_text}
                </p>
            </div>
            """, unsafe_allow_html=True)



            # --- 🪸 Chart container ---
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                plot_biodiversity_pie(results_all)
            with col2:
                plot_species_abundance(results_all)
            st.markdown('</div>', unsafe_allow_html=True)






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
            
        st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        

elif st.session_state.page == "Detection":

    import base64
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from collections import Counter
    import math
    import pandas as pd
    from components.result_box import render_results_box
    from components.species_info import get_species_info
    from components.loading_overlay import show_loading_overlay
    from components.auto_scroll import auto_scroll_to_results

    # --- Load YOLO model once ---
    @st.cache_resource
    def load_detection_model():
        from ultralytics import YOLO
        return YOLO("models/detection_model.pt")

    detection_model = load_detection_model()

    # --- Cache inference results (YOLO batch processing) ---
    @st.cache_data(show_spinner=False)
    def run_yolo_batch(_model, uploaded_files):
        """Run YOLO once per uploaded image and return results."""
        batch_results = []
        for f in uploaded_files:
            image = Image.open(f).convert("RGB")
            img_np = np.array(image)
            result = _model.predict(img_np)[0]
            batch_results.append((f.name, image, result))
        return batch_results

    def get_result_image(_result):
        buffer = BytesIO()
        Image.fromarray(_result.plot()).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


    # --- File uploader ---
    uploaded_files = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"detection_upload_v{st.session_state.detection_nonce}"
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        prev_names = st.session_state.get("last_uploaded_detection", [])
        added_files = [f for f in current_names if f not in prev_names]

        if added_files:
            st.session_state.last_uploaded_detection = current_names
            st.session_state.scrolled_after_upload = False
            show_loading_overlay("Running Detection Model...", duration=1.0)

        # --- Run inference once per batch (cached) ---
        results_batch = run_yolo_batch(detection_model, uploaded_files)

        # --- Initialize session state ---
        if "detection_results" not in st.session_state:
            st.session_state.detection_results = []
            st.session_state.current_detection_index = 0

        st.session_state.detection_results = []
        results_data = []

        def get_top_detection(result, names_map):
            """Get top detection from YOLO results."""
            if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
                return None, None
            confs = result.boxes.conf.cpu().numpy()
            idx = confs.argmax()
            top_conf = float(confs[idx])
            cls_id = int(result.boxes.cls[idx])
            label = names_map.get(cls_id, f"class_{cls_id}")
            return label, top_conf * 100.0

        # --- Process detections ---
        for filename, image, result in results_batch:
            results_b64 = get_result_image(result)
            detected_species, conf_percent = get_top_detection(result, detection_model.names)
            info = get_species_info(detected_species)

            st.session_state.detection_results.append({
                "filename": filename,
                "image_b64": results_b64,
                "species": detected_species,
                "confidence": conf_percent,
                "info": info
            })

            # Add detections to table
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    results_data.append({
                        "Filename": filename,
                        "Class": detection_model.names[cls_id],
                        "Confidence (%)": round(conf * 100, 2),
                        "Bounding Box": [round(x, 2) for x in xyxy]
                    })
            else:
                results_data.append({
                    "Filename": filename,
                    "Class": "No animal detected",
                    "Confidence (%)": 0.0,
                    "Bounding Box": []
                })

        # === 🧭 Display one image result at a time ===
        if st.session_state.detection_results:
            idx = st.session_state.current_detection_index
            results_list = st.session_state.detection_results
            idx = max(0, min(idx, len(results_list) - 1))
            current = results_list[idx]

            # --- Header ---
            st.markdown(
                f"<h4 style='text-align:center; color:#1565C0;'>"
                f"Image {idx+1} of {len(results_list)}: {current['filename']}</h4>",
                unsafe_allow_html=True
            )

            # --- Render current result ---
            render_results_box(
                image_bytes=base64.b64decode(current["image_b64"]),
                species=current["species"] or "No objects detected",
                confidence_percent=current["confidence"] or 0.0,
                species_info=current["info"]
            )

            def go_prev():
                st.session_state.current_detection_index = max(
                    0, st.session_state.current_detection_index - 1
                )
                st.session_state.scroll_to_results = True

            def go_next():
                st.session_state.current_detection_index = min(
                    len(st.session_state.detection_results) - 1,
                    st.session_state.current_detection_index + 1,
                )
                st.session_state.scroll_to_results = True 

            # --- Navigation Arrows ---
            st.markdown("<br>", unsafe_allow_html=True)
            nav_prev, _, nav_next = st.columns([1, 4, 1])

            with nav_prev:
                st.button(
                    "Previous",
                    key=f"prev_{idx}",
                    use_container_width=True,
                    disabled=(idx == 0),
                    on_click=go_prev,
                )

            with nav_next:
                st.button(
                    "Next",
                    key=f"next_{idx}",
                    use_container_width=True,
                    disabled=(idx == len(results_list) - 1),
                    on_click=go_next,
                )
        # === Convert results to DataFrame ===
        df_results = pd.DataFrame(results_data)

        # === 🌿 Ecological Insights ===
        show_insights = st.checkbox("Show Ecological Insights", value=False)
        if show_insights:
            st.markdown("## 🌿 Ecological Insights")

            results_all = [{"species": c} for c in df_results["Class"].tolist()]
            diversity_score, status = calculate_ecosystem_health(results_all)

            species_present = [r.get("species").strip().lower() for r in results_all if r.get("species")]
            keystone_found = [s for s in species_present if s.capitalize() in KEYSTONE_SPECIES.keys()]
            keystone_count = len(set([s.capitalize() for s in keystone_found]))

            # --- Color tier ---
            if keystone_count >= 6:
                box_color = "#2e7d32"
                biodiversity_text = "🌿 Strong biodiversity — most key species present."
            elif keystone_count >= 3:
                box_color = "#fbc02d"
                biodiversity_text = "🪸 Moderate biodiversity — some key species detected."
            elif keystone_count >= 1:
                box_color = "#ef6c00"
                biodiversity_text = "⚠️ Limited biodiversity — few key species detected."
            else:
                box_color = "#c62828"
                biodiversity_text = "🚫 No key species identified."

            species_list = [r.get("species") for r in results_all if r.get("species")]
            species_counts = Counter(species_list)
            total = sum(species_counts.values())
            num_species = len(species_counts)
            evenness = (diversity_score * math.log(len(species_counts))) / math.log(num_species) if num_species > 1 else 0.0
            dominant_species = max(species_counts, key=species_counts.get) if species_counts else "None"
            dominant_percent = (species_counts[dominant_species] / total) * 100 if total else 0

            if diversity_score > 0.75 and evenness > 0.6:
                summary_text = "Stable ecosystem — balanced and resilient community structure."
            elif diversity_score > 0.4:
                summary_text = "Moderately stable ecosystem with partial species balance."
            else:
                summary_text = "Low stability — one or two species dominate the environment."

            st.markdown(f"""
            <div style='
                background:rgba(0, 24, 61, 0.85);
                border-radius:15px;
                border:2px solid {box_color};
                box-shadow:0 8px 25px rgba(0,0,0,0.4);
                backdrop-filter:blur(12px);
                padding:1.8rem;
                margin-bottom:1.5rem;
                color:#E3F2FD;
                text-align:center;
            '>
                <h3 style='color:#64B5F6;'>Ecosystem Overview</h3>
                <p style='font-size:1.05rem; color:#90CAF9; margin-bottom:0.8rem;'><b>{status}</b></p>
                <hr style='border:0; height:1px; background:rgba(144,202,249,0.4); margin:1rem 0;'>
                <p style='font-size:0.95rem; line-height:1.6; color:#BBDEFB;'>
                    <b>Shannon Index:</b> {diversity_score:.3f} • 
                    <b>Evenness:</b> {evenness:.3f} • 
                    <b>Indicator Species:</b> {keystone_count}/7<br>
                    <b>Dominant Species:</b> {dominant_species} ({dominant_percent:.1f}%)
                </p>
                <p style='font-size:0.95rem; color:#B3E5FC; margin-top:1.2rem; line-height:1.6;'>
                    {biodiversity_text}<br>{summary_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # --- Biodiversity Charts ---
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                plot_biodiversity_pie(results_all)
            with col2:
                plot_species_abundance(results_all)
            st.markdown('</div>', unsafe_allow_html=True)

        # === 🐚 Detection Results Table ===
        st.write("### 🐚 Detection Results")
        st.dataframe(df_results, use_container_width=True)

        # === CSV Download ===
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="detection_results.csv",
            mime="text/csv"
        )

        



        
elif st.session_state.page == "Metrics":

    st.markdown("""
            <div style="
                text-align:center;
                background: rgba(255,255,255,0.6);
                backdrop-filter: blur(10px);
                border-radius: 18px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 2rem;
                margin-top: 2rem;
            ">
                <h2 style="color:#0D47A1; font-weight:800;">Classification Model</h2>
                <p style="color:#04365C; line-height:1.6;">
                    The dataset contained <strong>10,500 images</strong>, divided into <strong>7 classes</strong> (1,200 images per class). 
                    A <strong>train–test split of 80/20</strong> was used, resulting in <strong>8,400 training images</strong> and <strong>2,100 test images</strong>. 
                    An <strong>EfficientNet-v2-s</strong> model was trained for <strong>10 epochs</strong>, achieving a <strong>94.33% accuracy</strong> on the test set.
                </p>
             <div>
            """, unsafe_allow_html=True)
    st.markdown("##")  
    accuracy = 0.94
    macro_f1 = 0.94
    cm_image = Image.open("images/Confusion_matrix.png") 
    report = {
                "Eel": {"Precision": 0.98, "Recall": 0.99, "F1-score": 0.98, "Support": 300},
                "Scallop": {"Precision": 0.90, "Recall": 0.97, "F1-score": 0.93, "Support": 300},
                "Crab": {"Precision": 0.92, "Recall": 0.91, "F1-score": 0.91, "Support": 300},
                "Flatfish": {"Precision": 0.93, "Recall": 0.95, "F1-score": 0.94, "Support": 300},
                "Roundfish": {"Precision": 0.94, "Recall": 0.88, "F1-score": 0.91, "Support": 300},
                "Skate": {"Precision": 0.98, "Recall": 0.98, "F1-score": 0.98, "Support": 300},
                "Whelk": {"Precision": 0.96, "Recall": 0.93, "F1-score": 0.95, "Support": 300},
            }

    df_metrics = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<h2 style="color:#0D47A1; font-weight:800; text-align:center;">Confusion Matrix</h2>', unsafe_allow_html=True)
        st.image(cm_image, use_column_width=True)
    with col2:
        st.markdown('<h2 style="color:#0D47A1; font-weight:800; text-align:center;">Per-Class Classification Metrics</h2>', unsafe_allow_html=True)
        st.dataframe(df_metrics[["Class", "Precision", "Recall", "F1-score", "Support"]], use_container_width=True)
    
    st.markdown("""
            <div style="
                text-align:center;
                background: rgba(255,255,255,0.6);
                backdrop-filter: blur(10px);
                border-radius: 18px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 2rem;
                margin-top: 2rem;
            ">
                <h2 style="color:#0D47A1; font-weight:800;">Detection Model</h2>
                <p style="color:#04365C; line-height:1.6;">
                    The dataset contained <strong>2,759 images</strong> that were labelled and included the coordinates of bounding boxes. 
                    A <strong>YOLO11m</strong> model was trained for <strong>100 epochs</strong>, achieving a <strong>mAP of 90.1%</strong> on the test set
                </p>
            </div>
            """, unsafe_allow_html=True)
    df_detection_metrics = pd.DataFrame({
    "Class": ["Crab", "Eel", "Flatfish", "Roundfish", "Scallop", "Skate", "Whelk"],
    "Precision": [0.969, 0.802, 0.894, 0.709, 0.787, 0.924, 0.725],
    "Recall": [0.800, 0.941, 0.972, 0.737, 0.856, 0.934, 0.897],
    "mAP@50": [0.901, 0.926, 0.965, 0.776, 0.883, 0.957, 0.892],
    "mAP@50–95": [0.585, 0.711, 0.649, 0.547, 0.669, 0.774, 0.549]
})


    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<h2 style="color:#0D47A1; font-weight:800; text-align:center;">Detection Model Metrics</h2>', unsafe_allow_html=True)
        st.dataframe(df_detection_metrics.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "mAP@50": "{:.3f}",
                "mAP@50–95": "{:.3f}"
            }), use_container_width=True)
    with col2:
        st.markdown('<h2 style="color:#0D47A1; font-weight:800; text-align:center;">Bar Chart</h2>', unsafe_allow_html=True)
        df_plot = df_detection_metrics.melt(
        id_vars="Class",
        value_vars=["Precision", "Recall", "mAP@50"],
        var_name="Metric",
        value_name="Score"
    )

        # Plot
        plt.figure(figsize=(10,5))
        sns.barplot(x="Class", y="Score", hue="Metric", data=df_plot)
        plt.ylim(0,1)
        plt.title("Per-Class Detection Performance", fontsize=14, color="#0D47A1", fontweight="bold")
        plt.xlabel("Species")
        plt.ylabel("Score")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

