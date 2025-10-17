
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
from components.backgrounds.coral_classification import render_coral_scene
from components.backgrounds.deep_detection import render_deep_sea


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


with open("app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    description = "Discover the ocean's mysteries. Classify benthic species in underwater images with AI-powered recognition."
elif st.session_state.page == "Classification":
    render_coral_scene()
    description = (
        "Use this model to identify marine life in underwater photos. "
        "It classifies images into seven benthic species: crab, eel, whelk, scallop, flatfish, roundfish, and skate."
        
    )
elif st.session_state.page == "Detection":
    render_deep_sea()
    description = (
        "Use this model to locate and label marine species in underwater images by "
        "drawing bounding boxes around detected organisms and identifying their species."
    )
elif st.session_state.page == "Metrics":
    render_ocean_home()
    description = ("Explore model accuracy, precision, and performance results."
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





# ======================================================
# 🧭 NAVIGATION BAR (Home / Classification / Detection / Metrics)
# ======================================================

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Tab bar layout ---
st.markdown('<div class="tab-bar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
with col2:
    if st.button("Classification", use_container_width=True):
        st.session_state.page = "Classification"
        st.rerun()
with col3:
    if st.button("Detection", use_container_width=True):
        st.session_state.page = "Detection"
        st.rerun()
with col4:
    if st.button("Metrics", use_container_width=True):  # ✅ add consistent width
        st.session_state.page = "Metrics"
        st.rerun()

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
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                padding: 2.5rem 3rem;
                max-width: 800px;
                text-align: center;
                color: #04365c;
            ">
                <h1 style="font-size: 2.6rem; font-weight: 800; color: #0D47A1; margin-bottom: 0.4rem;">
                    Welcome to AMplify AI 🌊
                </h1>
                <p style="font-size: 1rem; color: #1565C0; margin-bottom: 1.8rem;">
                    <b>Created by:</b> Ariana Robinson &amp; Megan Timmes<br>
                    <span style="font-size: 0.95rem; color: #04365c;">
                        William &amp; Mary AI Case-a-thon — Fall 2025
                    </span>
                </p>
                <p style="font-size: 1.2rem; line-height: 1.7;">
                    We offer <b>2</b> fine-tuned deep learning models for marine life
                    <b>classification</b> and <b>detection</b>.<br><br>
                    Use the <b>Classification</b> model to identify benthic species in your photos,
                    or the <b>Detection</b> model to locate and label organisms in full underwater scenes.
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
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
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
            content: "Upload Underwater Image";
            font-size: 1.6rem;
            font-weight: 700;
            color: #0D47A1;
            display: block;
            margin-bottom: 1rem;
        }

        /* === REBUILD BROWSE BUTTON === */
        [data-testid="stFileUploader"] section div div div button {
            appearance: none !important;
            display: inline-block !important;
            border: none !important;
            outline: none !important;
            background: linear-gradient(to right, #1565C0, #1E88E5) !important;
            color: white !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 14px 32px !important;
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
                    An <strong>EfficientNet-B0</strong> model was trained for <strong>15 epochs</strong>, achieving a <strong>91.14% accuracy</strong> on the test set.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
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
                    The dataset contained <strong>2,759 images</strong>, divided into <strong>7 classes</strong>. 
                    A <strong>train–test split of 80/20</strong> was used, resulting in <strong>8,400 training images</strong> and <strong>2,100 test images</strong>. 
                    A <strong>YOLO11m</strong> model was trained for <strong>100 epochs</strong>, achieving a <strong>mAP of 90%</strong> on the test set
                </p>
            </div>
            """, unsafe_allow_html=True)


