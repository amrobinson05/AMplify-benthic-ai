import base64
import streamlit as st

def render_results_box(image_bytes, species, confidence_percent, species_info=None):
    """Render the result card with the uploaded (or annotated) image + species details."""
    b64_img = base64.b64encode(image_bytes).decode()

    # Safely handle missing info dictionary
    habitat = species_info.get("Habitat", "N/A") if species_info else "N/A"
    depth = species_info.get("Depth Range", "N/A") if species_info else "N/A"
    fun_fact = species_info.get("Fun Fact", "N/A") if species_info else "N/A"
    desc = species_info.get("Description", "N/A") if species_info else "N/A"

    st.markdown(f"""
<div class="results-box" style="
    background: rgba(255,255,255,0.6);
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    padding: 2rem;
    margin-top: 2rem;
">
    <div style="display:flex; gap:2rem; align-items:center; justify-content:center; flex-wrap:wrap;">
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
            <p style="margin:0.3rem 0;"><b>Habitat:</b> {habitat}</p>
            <p style="margin:0.3rem 0;"><b>Depth Range:</b> {depth}</p>
            <p style="margin:0.3rem 0;"><b>Fun Fact:</b> {fun_fact}</p>
            <p style="margin:0.3rem 0;"><b>Description:</b> {desc}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
