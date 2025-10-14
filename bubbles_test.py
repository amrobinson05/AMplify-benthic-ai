import streamlit as st

st.set_page_config(page_title="Bubble Test", layout="wide")

# Bubble CSS and animation
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  height: 100%;
  margin: 0;
  padding: 0;
  background: linear-gradient(to bottom, #edfffa 0%, #31c5d6 100%);
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
  0% {
    transform: translateY(100vh) scale(0.5);
    opacity: 0.8;
  }
  100% {
    transform: translateY(-10vh) scale(1.2);
    opacity: 0;
  }
}
</style>

<div class="bubble-container">
""" + "\n".join([
    f"<div class='bubble' style='left:{i*3.3}%; width:{10 + (i%5)*20}px; height:{10 + (i%5)*20}px; animation-duration:{5 + (i%6)*4}s; animation-delay:{i*0.7}s;'></div>"
    for i in range(40)
]) + "</div>",
unsafe_allow_html=True)

st.title("🫧 Bubble Animation Test")
st.write("If you see bubbles floating upward, it works! 🌊")
