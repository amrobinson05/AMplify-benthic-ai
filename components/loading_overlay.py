import streamlit as st, base64, time, os

def show_loading_overlay(
    message="Analyzing Marine Life...",
    duration=1.0,
    image_path="images/starfish.png",
    bg_color="rgba(21,101,192,0.55)"
):
    """Displays the animated loading overlay with a spinner and progress bar."""

    if not os.path.exists(image_path):
        st.error(f"Missing overlay image: {image_path}")
        return

    starfish_b64 = base64.b64encode(open(image_path, "rb").read()).decode()

    st.markdown(f"""
    <style>
    #loading-overlay {{
        position:fixed; top:0; left:0; width:100vw; height:100vh;
        background:{bg_color};
        backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        z-index:2147483647;
    }}
    @keyframes spinStar {{0%{{transform:rotate(0deg);}}100%{{transform:rotate(360deg);}}}}
    </style>
    """, unsafe_allow_html=True)

    loading_placeholder = st.empty()
    for i in range(31):
        progress = i * (100 / 30)
        loading_placeholder.markdown(f"""
        <div id="loading-overlay">
            <img class="spinStar" src="data:image/png;base64,{starfish_b64}"
                style="width:120px; height:auto; animation:spinStar 2.5s linear infinite;
                        filter:drop-shadow(0 0 10px rgba(255,255,255,0.8));"/>
            <p style="font-size:1.8rem; font-weight:700; margin-top:1rem; color:white;">
                {message} {progress:.0f}%
            </p>
            <div style="margin-top:1.5rem; width:300px; height:12px;
                        background:rgba(255,255,255,0.3); border-radius:10px; overflow:hidden;">
                <div style="width:{progress}%; height:100%;
                    background:linear-gradient(to right,#64B5F6,#1565C0);
                    border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(duration / 30.0)
    loading_placeholder.empty()
