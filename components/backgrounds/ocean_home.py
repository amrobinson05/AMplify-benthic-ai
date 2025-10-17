import streamlit as st
import os, base64, random

def render_ocean_home():
    """Render tropical ocean background with bubbles, fish, and seagrass animation."""

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
      background: linear-gradient(
        to bottom,
        #e9fcff 0%,
        #bdf4f6 25%,
        #75d3e2 55%,
        #3bbad1 75%,
        #b0eac4 100%
      );
      overflow: hidden;
    }
    [data-testid="stAppViewContainer"], .main, .block-container {
      background-color: transparent !important;
    }

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

    for i in range(30):
        x = random.randint(0, width)
        y = random.randint(120, 300)
        size = random.randint(10, 25)
        color = random.choice(["#5cd57c", "#6de28a", "#79eaa0", "#8ff3b2"])
        kelp_balls.append(f"<circle cx='{x}' cy='{y}' r='{size/2}' fill='{color}' opacity='0.85' />")

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

        # ======================================================
    # FIX LAYERING (make all visuals behind UI)
    # ======================================================
    st.markdown("""
    <style>
    /* 🐠 Everything visual behind */
    .bubble-container,
    .species-container,
    #seagrass-svg {
        z-index: 0 !important;
    }

    /* 🧊 Streamlit main content (text boxes, titles, etc.) above */
    [data-testid="stAppViewContainer"] .main {
        position: relative !important;
        z-index: 10 !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stVerticalBlock"],
    [data-testid="stImage"] {
        position: relative !important;
        z-index: 20 !important;
    }

    /* Ensure overlay effects (like loading spinner) can go on top */
    #loading-overlay {
        z-index: 9999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

