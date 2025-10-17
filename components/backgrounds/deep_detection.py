import streamlit as st
import random

def render_deep_sea():
    """Render a deep-sea background for the Detection page."""

    # === Deep-sea gradient background ===
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: linear-gradient(
            to bottom,
            #00172d 0%,      /* dark navy surface */
            #002b45 30%,     /* deep-blue midwater */
            #003e5e 65%,     /* darker gradient */
            #001a28 100%     /* abyssal trench */
        );
    }

    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: transparent !important;
    }

    /* === Gentle light rays === */
    .light-ray {
        position: fixed;
        top: -20vh;
        width: 3px;
        height: 140vh;
        background: linear-gradient(to bottom, rgba(255,255,255,0.2), rgba(255,255,255,0));
        filter: blur(1px);
        opacity: 0.2;
        animation: swayLight 8s ease-in-out infinite alternate;
        z-index: 0;
        pointer-events: none;
    }
    @keyframes swayLight {
        0% { transform: rotate(-4deg) translateX(-10px); }
        100% { transform: rotate(4deg) translateX(10px); }
    }

    /* === Bioluminescent particles === */
    .biolume {
        position: absolute;
        background: radial-gradient(circle, rgba(0,255,255,0.9) 0%, rgba(0,255,255,0) 70%);
        border-radius: 50%;
        opacity: 0.8;
        animation: floatGlow 12s linear infinite;
    }
    @keyframes floatGlow {
        0% { transform: translateY(100vh) scale(0.5); opacity: 0.9; }
        100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
    }

    /* === Ocean floor silhouettes === */
    .trench {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw;
        height: 30vh;
        background: linear-gradient(to top, #001019, transparent);
        z-index: 1;
    }

    /* Keep UI above everything */
    [data-testid="stAppViewContainer"] .main {
        position: relative;
        z-index: 10 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === Add light rays ===
    rays_html = ""
    for i in range(8):
        left = random.randint(0, 100)
        delay = random.uniform(0, 3)
        rays_html += f"<div class='light-ray' style='left:{left}vw; animation-delay:{delay}s;'></div>"
    st.markdown(rays_html, unsafe_allow_html=True)

    # === Add bioluminescent glow particles ===
    glows_html = "<div style='position:fixed;top:0;left:0;width:100vw;height:100vh;overflow:hidden;pointer-events:none;z-index:0;'>"
    for i in range(50):
        x = random.randint(0, 100)
        size = random.randint(3, 8)
        duration = random.uniform(10, 18)
        delay = random.uniform(0, 6)
        glow = f"<div class='biolume' style='left:{x}vw;width:{size}px;height:{size}px;animation-duration:{duration}s;animation-delay:{delay}s;'></div>"
        glows_html += glow
    glows_html += "</div>"
    st.markdown(glows_html, unsafe_allow_html=True)

    # === Add trench overlay ===
    st.markdown("<div class='trench'></div>", unsafe_allow_html=True)
