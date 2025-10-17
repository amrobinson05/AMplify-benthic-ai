import streamlit as st
import random, base64, os

def render_deep_sea():
    """Render a deep-sea trench scene with corals, marine snow, glowing orbs, jellyfish, and crab animation."""

    # === BACKGROUND GRADIENT ===
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
    margin: 0;
    padding: 0;
    background: linear-gradient(
        to bottom,
        #02203b 0%,     /* lighter navy blue near surface */
        #023e59 50%,    /* teal midwater */
        #013047 100%    /* deep ocean */
    );
    overflow: hidden;
}
[data-testid="stAppViewContainer"], .main, .block-container {
    background-color: transparent !important;
}

#trench-svg {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100vw;
    height: 300px;
    z-index: 0;
    pointer-events: none;
}

@keyframes shimmer {
    0% { filter: brightness(1); }
    50% { filter: brightness(1.05); }
    100% { filter: brightness(1); }
}
</style>
""", unsafe_allow_html=True)


    # === ROCK RIDGES ===
    width = 1200
    rocks_layer1, rocks_layer2 = [], []

    for i in range(6):
        x1 = i * 200
        y1 = random.randint(240, 300)
        x2 = x1 + 250
        y2 = random.randint(260, 320)
        rocks_layer1.append(f"{x1},{y1} {x2},{y2}")

    for i in range(8):
        x1 = i * 160
        y1 = random.randint(300, 350)
        x2 = x1 + 180
        y2 = random.randint(340, 370)
        rocks_layer2.append(f"{x1},{y1} {x2},{y2}")

    layer1_points = " ".join(rocks_layer1)
    layer2_points = " ".join(rocks_layer2)

    trench_svg = f"""
    <svg id="trench-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400" preserveAspectRatio="none">
        <defs>
            <linearGradient id="rockGradient1" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#001822"/>
                <stop offset="100%" stop-color="#000d12"/>
            </linearGradient>
            <linearGradient id="rockGradient2" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#001222"/>
                <stop offset="100%" stop-color="#000810"/>
            </linearGradient>
        </defs>
        <polygon points="0,300 {layer1_points} 1200,300 1200,400 0,400"
                 fill="url(#rockGradient1)" opacity="0.85"
                 style="animation: shimmer 10s ease-in-out infinite;" />
        <polygon points="0,340 {layer2_points} 1200,340 1200,400 0,400"
                 fill="url(#rockGradient2)" opacity="0.95" />
    </svg>
    """
    st.markdown(trench_svg, unsafe_allow_html=True)


    # === MARINE SNOW ===
    snow = []
    for i in range(50):
        left = random.randint(0, 100)
        size = random.uniform(2, 4)
        delay = random.uniform(0, 8)
        duration = random.uniform(8, 20)
        opacity = random.uniform(0.3, 0.8)
        snow.append(
            f"<div class='marine-snow' style='left:{left}vw; width:{size}px; height:{size}px; opacity:{opacity}; "
            f"animation-delay:{delay}s; animation-duration:{duration}s;'></div>"
        )

    # === BIOLUMINESCENT ORBS ===
    orbs = []
    for i in range(12):
        left = random.randint(0, 100)
        top = random.randint(10, 90)
        size = random.uniform(10, 35)
        color = random.choice(["rgba(0,255,200,0.2)", "rgba(0,200,255,0.25)", "rgba(0,180,255,0.3)"])
        delay = random.uniform(0, 6)
        duration = random.uniform(3, 8)
        orbs.append(
            f"<div class='biolume' style='left:{left}vw; top:{top}vh; width:{size}px; height:{size}px; "
            f"background:{color}; animation-delay:{delay}s; animation-duration:{duration}s;'></div>"
        )

    # === CSS for particles + jellyfish + crab ===
    st.markdown("""
    <style>
    .marine-snow {
        position: fixed;
        top: -20px;
        border-radius: 50%;
        background: rgba(255,255,255,0.7);
        animation: fallSnow linear infinite;
        z-index: 2;
        pointer-events: none;
    }
    @keyframes fallSnow {
        0% { transform: translateY(0px) scale(0.9); opacity: 1; }
        100% { transform: translateY(100vh) scale(1.1); opacity: 0; }
    }

    .biolume {
        position: fixed;
        border-radius: 50%;
        filter: blur(8px);
        animation: glowPulse ease-in-out infinite;
        z-index: 1;
        pointer-events: none;
    }
    @keyframes glowPulse {
        0%,100% { opacity: 0.15; transform: scale(0.9); }
        50% { opacity: 0.6; transform: scale(1.3); }
    }

    /* === Glowing Floating Jellyfish === */
    @keyframes jellyFloat {
        0% { transform: translateY(0px) scale(1); opacity: 0.9; }
        50% { transform: translateY(-60px) scale(1.05); opacity: 1; }
        100% { transform: translateY(0px) scale(1); opacity: 0.9; }
    }
    @keyframes jellyGlow {
        0%,100% { filter: drop-shadow(0 0 12px rgba(0,255,255,0.5)) brightness(1.1); }
        50% { filter: drop-shadow(0 0 25px rgba(0,255,255,0.9)) brightness(1.5); }
    }
    .jellyfish {
        position: fixed;
        width: 160px;
        height: auto;
        opacity: 0.9;
        animation: jellyFloat 8s ease-in-out infinite, jellyGlow 3s ease-in-out infinite;
        z-index: 2;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

       # === RENDER IMAGES ===
    def img_b64(filename):
        with open(os.path.join("images", filename), "rb") as f:
            return base64.b64encode(f.read()).decode()

    jelly_b64 = img_b64("jellyfish.png")

    # 🪼 Spread jellyfish farther apart across the screen
    jelly_html = f"""
    <div id="jellyfish-layer" style="z-index: 0; position: fixed; width:100%; height:100%; pointer-events:none;">
        <img class="jellyfish" src="data:image/png;base64,{jelly_b64}"
             style="left:8vw; bottom:18vh; animation-delay:0s;">
        <img class="jellyfish" src="data:image/png;base64,{jelly_b64}"
             style="left:45vw; bottom:25vh; animation-delay:2s;">
        <img class="jellyfish" src="data:image/png;base64,{jelly_b64}"
             style="left:78vw; bottom:22vh; animation-delay:4s;">
    </div>
    """


    # === Render particle + creature layers (in proper order) ===
    # 1️⃣ background effects (snow + orbs)
    st.markdown(f"<div id='marine-snow-container'>{''.join(snow)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div id='biolume-container'>{''.join(orbs)}</div>", unsafe_allow_html=True)

    # 2️⃣ jellyfish BELOW description box (lower z-index)
    st.markdown(jelly_html, unsafe_allow_html=True)


    # extra style to ensure page text/UI appears above all background
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] .main {
        position: relative !important;
        z-index: 5 !important; /* page content always above jellyfish */
    }
    #jellyfish-layer, .jellyfish {
        z-index: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
