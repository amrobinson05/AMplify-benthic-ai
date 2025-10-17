import streamlit as st
import random

def render_coral_scene():
    """Render a shallow coral reef background for the Classification page."""
    
    # === Coral reef gradient background ===
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: linear-gradient(
            to bottom,
            #b8faff 0%,       /* bright turquoise surface */
            #7be2e0 30%,      /* tropical blue */
            #3bbdc4 65%,      /* teal-blue midwater */
            #2e8894 85%,      /* deeper coral shadow */
            #265974 100%      /* ocean floor */
        );
    }

    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: transparent !important;
    }

    /* === Floating coral silhouettes === */
    .coral-scene {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }

    .coral {
        position: absolute;
        bottom: 0;
        opacity: 0.9;
        filter: drop-shadow(0px 3px 5px rgba(0,0,0,0.3));
        animation: swayCoral 6s ease-in-out infinite alternate;
    }

    @keyframes swayCoral {
        0% { transform: rotate(-2deg); }
        100% { transform: rotate(2deg); }
    }

    /* === Floating plankton sparkles === */
    .sparkle {
        position: absolute;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 50%;
        width: 4px;
        height: 4px;
        animation: floatSparkle 10s linear infinite;
        opacity: 0.6;
    }

    @keyframes floatSparkle {
        0% { transform: translateY(100vh) scale(0.6); opacity: 0.8; }
        100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
    }

    /* Keep all UI above coral */
    [data-testid="stAppViewContainer"] .main {
        position: relative;
        z-index: 10 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === Random sparkles ===
    sparkles_html = "<div class='coral-scene'>"
    for i in range(60):
        left = random.randint(0, 100)
        size = random.randint(2, 5)
        duration = random.uniform(8, 14)
        delay = random.uniform(0, 5)
        sparkle = f"<div class='sparkle' style='left:{left}vw; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;'></div>"
        sparkles_html += sparkle
    sparkles_html += "</div>"
    st.markdown(sparkles_html, unsafe_allow_html=True)

    # === Coral SVGs (colorful seafloor silhouettes) ===
    coral_svg = """
    <svg class="coral-scene" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400" preserveAspectRatio="none">
        <path d="M0,300 C200,270 400,340 600,320 C800,300 1000,340 1200,310 L1200,400 L0,400 Z"
              fill="#ffb6b9" opacity="0.8" />
        <path d="M0,330 C250,310 450,380 700,360 C950,340 1200,390 1200,390 L1200,400 L0,400 Z"
              fill="#fcd5ce" opacity="0.9" />
        <path d="M0,350 C150,330 300,400 500,380 C700,360 1000,400 1200,370 L1200,400 L0,400 Z"
              fill="#f9bec7" opacity="0.7" />
    </svg>
    """
    st.markdown(coral_svg, unsafe_allow_html=True)
