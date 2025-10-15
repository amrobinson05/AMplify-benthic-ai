
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


# PAGE CONFIG
st.set_page_config(
    page_title="Benthic AI Dashboard",
    page_icon="🌊",
    layout="wide"
)

# ✅ Initialize session state once
if "page" not in st.session_state:
    st.session_state.page = "Home"

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
st.markdown("""
<style>
/* Lower the animation layers */
.bubble-container,
#seagrass-svg,
.species-container {
    z-index: 0 !important;
}

/* Keep Streamlit UI (buttons, uploader, images, etc.) above everything */
[data-testid="stAppViewContainer"] .main {
    position: relative;
    z-index: 10 !important;
}

/* Also ensure file uploader and output text stay above visuals */
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



st.markdown(
    """
    <style>
    .gradient-text {
        font-size: 70px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #0D47A1, #1565C0, #64B5F6);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>

    <div style='text-align:center;'>
        <span class='gradient-text'>Benthic Species Identifier</span>
        <h5 style='text-align:center; color:#37474F;'>
            Discover the ocean's mysteries. Upload a photo to instantly identify a benthic species<br>
            with AI-powered recognition.
        </h5>
    </div>
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

st.markdown('<div class="tab-bar">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
with col2:
    if st.button("Classification", use_container_width=True):
        st.session_state.page = "Classification"
with col3:
    if st.button("Detection", use_container_width=True):
        st.session_state.page = "Detection"
st.markdown('</div>', unsafe_allow_html=True)


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
    model.load_state_dict(torch.load("benthic_model.pth", map_location=torch.device("cpu")))
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
        <div style='text-align:center; margin-top:4rem;'>
            <h1 style='font-size:3rem; font-weight:800; color:#0D47A1;'>Welcome to Benthic AI 🌊</h1>
            <p style='font-size:1.2rem; color:#04365c;'>
                Explore the ocean’s mysteries using AI-powered marine classification and detection.<br><br>
                Choose <b>Classification</b> to identify species in your photos, or <b>Detection</b> to analyze objects in underwater scenes.
            </p>
        </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Classification":
    # 🐚 Your original classification logic
    st.markdown("### 🐚 Classification Page")

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

        loading_placeholder = st.empty()
        loading_placeholder.markdown(loading_html, unsafe_allow_html=True)

        # 3️⃣ Animate loading bar (3 seconds)
        progress_placeholder = st.empty()
        for i in range(31):
            progress_placeholder.markdown(f"""
            <script>
            const bar = document.getElementById('load-bar');
            if (bar) bar.style.width = '{i * (100/30)}%';
            </script>
            """, unsafe_allow_html=True)
            time.sleep(0.1)

        results_all = []  # store results for all images

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
            if i < 2:
                b64_img = base64.b64encode(img_bytes).decode()  # use the same bytes
                st.markdown(
                    f"""
                    <div class="results-box">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <p style="font-size:1.4rem; font-weight:800; color:#0D47A1; margin-bottom:0;">Predicted Species</p>
                                <p style="font-size:1.2rem; font-weight:700; color:#04365c; margin-top:0;">{species}</p>
                            </div>
                            <div style="text-align:right;">
                                <p style="font-size:1.4rem; font-weight:800; color:#0D47A1; margin-bottom:0;">Confidence</p>
                                <p style="font-size:1.2rem; font-weight:700; color:#04365c; margin-top:0;">{confidence_percent:.1f}%</p>
                                <div style="width:200px; height:10px; background:rgba(0,0,0,0.1); border-radius:8px; overflow:hidden;">
                                    <div style="width:{confidence_percent}%; height:100%; background:linear-gradient(to right,#3b82f6,#60a5fa); border-radius:8px;"></div>
                                </div>
                            </div>
                        </div>
                        <div style="text-align:center; margin-top:2rem;">
                            <img src="data:image/png;base64,{b64_img}"
                                style="width:300px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.15);"/>
                            <p style="margin-top:10px; font-weight:600; color:#04365c;">
                                Predicted: {species} ({confidence_percent:.1f}%)
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
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

        st.write("### 🐚 Classification Results")
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
        # Remove overlay after loading
        loading_placeholder.empty()
        progress_placeholder.empty()

        import streamlit.components.v1 as components

        # === SMOOTH AUTO-SCROLL TO RESULTS (clean version) ===
        components.html("""
        <script>
        (function(){
        const doc = window.parent.document;
        const target = doc.querySelector('#results-anchor') || doc.querySelector('.results-box');
        if (!target) return;
        setTimeout(() => {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Optional: flash border glow for visibility
            target.style.boxShadow = '0 0 20px rgba(21,101,192,0.6)';
            setTimeout(() => { target.style.boxShadow = 'none'; }, 900);
        }, 100);
        })();
        </script>
        """, height=0)




    else:
        st.info("⬆️ Upload an image to begin classification.")
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
    st.markdown("### 🔎 Detection Page")
    st.info("This page will use the Detection LLM once it's trained.")
    st.markdown("""
        <div style='padding: 2rem; background: rgba(255,255,255,0.6);
                    border-radius: 10px; text-align:center;'>
            <h3>🚧 Detection model coming soon!</h3>
            <p>Once your detection model is ready, you'll be able to upload underwater 
            images here for bounding-box detection and segmentation of marine species.</p>
        </div>
    """, unsafe_allow_html=True)
