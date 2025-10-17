import base64
import streamlit as st
import streamlit.components.v1 as components

def render_results_box(image_bytes, species, confidence_percent, species_info=None):
    """Render the result card with a wider rectangular image and right-side info boxes."""
    b64_img = base64.b64encode(image_bytes).decode()

    # Safely handle missing info dictionary
    habitat = species_info.get("Habitat", "N/A") if species_info else "N/A"
    depth = species_info.get("Depth Range", "N/A") if species_info else "N/A"
    fun_fact = species_info.get("Fun Fact", "N/A") if species_info else "N/A"
    desc = species_info.get("Description", "N/A") if species_info else "N/A"

    html = f"""
    <div style="
        background: rgba(255,255,255,0.7);
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        padding: 2.5rem;
        margin-top: 2rem;
        display: flex;
        flex-wrap: wrap;
        gap: 2.8rem;
        align-items: flex-start;
        justify-content: center;
        font-family: 'Inter', sans-serif;
    ">
        <!-- LEFT COLUMN -->
        <div style="flex: 1.2; min-width: 420px; text-align: center;">
            <h2 style="color:#0D47A1; font-weight:800; margin-bottom:1rem; font-size:1.8rem;">
                {species}
            </h2>

            <img src="data:image/png;base64,{b64_img}" 
                 style="width:520px; height:350px; border-radius:14px;
                        box-shadow:0 6px 14px rgba(0,0,0,0.25);" />

            <div style="margin-top:1.5rem; background:rgba(255,255,255,0.9);
                        border-radius:12px; padding:1rem 1.4rem;
                        box-shadow:0 4px 10px rgba(0,0,0,0.12); text-align:left;
                        width:520px; margin-left:auto; margin-right:auto;">
                <p style="margin:0; font-size:1.05rem; color:#04365c; line-height:1.6;">
                    <b>Description:</b> {desc}
                </p>
            </div>
        </div>

        <!-- RIGHT COLUMN -->
        <div style="flex: 1; min-width: 320px;">
            <!-- Accuracy Bar -->
            <div style="margin-bottom: 2rem;">
                <p style="font-weight:700; color:#04365c; margin-bottom:8px; font-size:1.05rem;">
                    Model Confidence
                </p>
                <div style="width: 520px; height: 10px; background:rgba(0,0,0,0.1);
                            border-radius:10px; overflow:hidden;">
                    <div style="width:{confidence_percent}%; height:100%;
                                background:linear-gradient(to right,#3b82f6,#1565C0);
                                border-radius:10px;">
                    </div>
                </div>
                <p style="font-size:1rem; color:#04365c; margin-top:6px;">
                    {confidence_percent:.1f}% confidence
                </p>
            </div>

            <!-- Info Boxes -->
            <div style="display:flex; flex-direction:column; gap:1.2rem;">
                <!-- Habitat -->
                <div style="background:rgba(21,101,192,0.12); border-left:6px solid #1565C0;
                            border-radius:12px; padding:1rem 1.2rem;
                            box-shadow:0 4px 10px rgba(0,0,0,0.1);
                            font-size:1.05rem; line-height:1.5; color:#0D47A1;">
                    <b>Habitat:</b><br>{habitat}
                </div>

                <!-- Depth Range -->
                <div style="background:rgba(13,71,161,0.15); border-left:6px solid #0D47A1;
                            border-radius:12px; padding:1rem 1.2rem;
                            box-shadow:0 4px 10px rgba(0,0,0,0.1);
                            font-size:1.05rem; line-height:1.5; color:#083b79;">
                    <b>Depth Range:</b><br>{depth}
                </div>

                <!-- Fun Fact -->
                <div style="background:rgba(100,181,246,0.15); border-left:6px solid #42A5F5;
                            border-radius:12px; padding:1rem 1.2rem;
                            box-shadow:0 4px 10px rgba(0,0,0,0.1);
                            font-size:1.05rem; line-height:1.5; color:#084c93;">
                    <b>Fun Fact:</b><br>{fun_fact}
                </div>
            </div>
        </div>
    </div>
    """
    # Remove Streamlit's default padding/background around the HTML component
    # 🔧 Remove the outer Streamlit box and iframe border completely
    st.markdown("""
    <style>
    /* Remove extra white/gray box Streamlit adds around components.html */
    iframe[srcdoc] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Remove the padding/margin Streamlit adds around that iframe */
    [data-testid="stVerticalBlock"] > div,
    [data-testid="stHorizontalBlock"] > div,
    [data-testid="stComponent"] {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Make sure the entire area around it blends with your ocean background */
    .block-container {
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


    components.html(html, height=720, scrolling=False)
