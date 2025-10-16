import streamlit as st, base64

def render_result_box(
    image_b64: str,
    title: str,
    subtitle: str,
    right_content: list,
    progress_value: float = None
):
    """
    Reusable result box (used by both classification & detection).
    right_content: list of HTML strings (each line or <p>).
    """
    progress_html = ""
    if progress_value is not None:
        progress_html = f"""
        <div style="width:200px; height:10px; background:rgba(0,0,0,0.1);
                    border-radius:8px; overflow:hidden; margin:auto;">
            <div style="width:{progress_value:.1f}%; height:100%;
                background:linear-gradient(to right,#3b82f6,#60a5fa);
                border-radius:8px;"></div>
        </div>"""

    st.markdown(f"""
    <div class="results-box" style="
        background: rgba(255,255,255,0.6);
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        padding: 2rem;
        margin-top: 2rem;
    ">
        <div style="display:flex; gap:2rem; align-items:center; justify-content:center; flex-wrap:wrap;">
            <!-- Left: Image -->
            <div style="flex:1; min-width:280px; text-align:center;">
                <img src="data:image/png;base64,{image_b64}"
                    style="width:300px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.15);" />
                <p style="margin-top:10px; font-weight:600; color:#04365c;">{subtitle}</p>
                {progress_html}
            </div>
            <div style="flex:1; min-width:300px;">
                <h3 style="color:#0D47A1; font-weight:800; margin-bottom:0.5rem;">{title}</h3>
                {"".join(right_content)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
