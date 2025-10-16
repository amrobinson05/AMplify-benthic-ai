import streamlit as st
import base64

def render_results_box(image_bytes, species, confidence_percent, species_info=None):
    """Reusable results box for displaying classification results."""
    b64_img = base64.b64encode(image_bytes).decode()
    info = species_info.get(species, None) if species_info else None

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
                <img src="data:image/png;base64,{b64_img}" 
                    style="width:300px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.15);" />
                <p style="margin-top:10px; font-weight:600; color:#04365c;">
                    Predicted: {species} ({confidence_percent:.1f}%)
                </p>
                <div style="width:200px; height:10px; background:rgba(0,0,0,0.1); border-radius:8px; overflow:hidden; margin:auto;">
                    <div style="width:{confidence_percent}%; height:100%;
                        background:linear-gradient(to right,#3b82f6,#60a5fa);
                        border-radius:8px;">
                    </div>
                </div>
            </div>
            <div style="flex:1; min-width:300px;">
                <h3 style="color:#0D47A1; font-weight:800; margin-bottom:0.5rem;">
                    🌊 About the {species}
                </h3>
                <p style="margin:0.3rem 0;"><b>Habitat:</b> {info['Habitat'] if info else 'N/A'}</p>
                <p style="margin:0.3rem 0;"><b>Depth Range:</b> {info['Depth Range'] if info else 'N/A'}</p>
                <p style="margin:0.3rem 0;"><b>Fun Fact:</b> {info['Fun Fact'] if info else 'N/A'}</p>
                <p style="margin:0.3rem 0;"><b>Description:</b> {info['Description'] if info else 'N/A'}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
