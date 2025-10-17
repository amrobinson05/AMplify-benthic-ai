
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

def clear_uploaded_files():
    """Reset uploaded files when switching models."""
    for key in ["classification_upload", "detection_upload"]:
        if key in st.session_state:
            st.session_state[key] = None  # fully reset uploader state


# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)

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

    # Track last uploaded files so we can detect new uploads
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = []


    import time

    if uploaded_files:
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
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != [f.name for f in uploaded_files]:
            st.session_state.last_uploaded = [f.name for f in uploaded_files]
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
            if i < 1:
                from components.species_info import get_species_info
                from components.auto_scroll import auto_scroll_to_results

                st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)
                auto_scroll_to_results()
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
        

elif st.session_state.page == "Detection":

    @st.cache_resource
    def load_detection_model():
        from ultralytics import YOLO
        return YOLO("models/detection_model.pt")  # path to your model

    detection_model = load_detection_model()
    # --- In Detection page ---
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"detection_upload_v{st.session_state.detection_nonce}"
    )


    if uploaded_file:
        from components.loading_overlay import show_loading_overlay
        from components.result_box import render_results_box
        import base64
        from io import BytesIO
        from components.species_info import species_info
        from components.species_info import get_species_info

        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != [f.name for f in uploaded_file]:
            st.session_state.last_uploaded = [f.name for f in uploaded_file]
            from components.loading_overlay import show_loading_overlay
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
                from components.auto_scroll import auto_scroll_to_results
                st.markdown("<div id='results-anchor'></div>", unsafe_allow_html=True)
                auto_scroll_to_results()
                render_results_box(
                    image_bytes=base64.b64decode(results_b64),
                    species=detected_species or "No objects detected",
                    confidence_percent=conf_percent or 0.0,
                    species_info=info

                )
        

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

        # ✅ Only show summary if detections exist
        if not df_results.empty and "Class" in df_results.columns:
            summary_df = (
                df_results.groupby("Class")
                .size()
                .reset_index(name="Total Detections")
                .sort_values(by="Total Detections", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No detections found in uploaded images.")

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
    cm_image = Image.open("images/Confusion_matrix.png") 
    st.image(cm_image, caption="Normalized Confusion Matrix",  width=500)
    
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
                    A <strong>YOLO11m</strong> model was trained for <strong>100 epochs</strong>, achieving a <strong>mAP of 90%</strong> on the test set
                </p>
            </div>
            """, unsafe_allow_html=True)


