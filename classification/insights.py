import streamlit as st
import math
from collections import Counter

# --- Local project utilities ---
from utils.ecosystem import calculate_ecosystem_health
from utils.charts import plot_biodiversity_pie, plot_species_abundance


KEYSTONE_SPECIES = {
    "Scallop": "", "Roundfish": "", "Crab": "",
    "Whelk": "", "Skate": "", "Flatfish": "", "Eel": ""
}

def display_ecological_insights(results_all):
    """
    Render the entire 'Ecological Insights' section.
    Expects results_all = list of dicts with {species, confidence, probs}
    """

    show_insights = st.checkbox("Show Ecological Insights", value=False)
    if not show_insights:
        return

    # --- Title ---
    st.markdown("## 🌿 Ecological Insights")

    # --- Calculate ecosystem metrics ---
    diversity_score, status = calculate_ecosystem_health(results_all)
    species_list = [r.get("species") for r in results_all if r.get("species")]
    species_counts = Counter(species_list)
    total = sum(species_counts.values())
    num_species = len(species_counts)

    # --- Keystone species presence ---
    keystone_found = [s for s in species_list if s.capitalize() in KEYSTONE_SPECIES]
    keystone_count = len(set([s.capitalize() for s in keystone_found]))

    # --- Evenness (Pielou’s J) ---
    if num_species > 1:
        evenness = (diversity_score * math.log(len(species_counts))) / math.log(num_species)
    else:
        evenness = 0.0

    # --- Dominant species ---
    if species_counts:
        dominant_species = max(species_counts, key=species_counts.get)
        dominant_percent = (species_counts[dominant_species] / total) * 100
    else:
        dominant_species, dominant_percent = "None", 0

    # --- Biodiversity tier color ---
    if keystone_count >= 6:
        box_color = "#2e7d32"
        biodiversity_text = "🌿 Strong biodiversity — most key species present."
    elif keystone_count >= 3:
        box_color = "#fbc02d"
        biodiversity_text = "🪸 Moderate biodiversity — some key species detected."
    elif keystone_count >= 1:
        box_color = "#ef6c00"
        biodiversity_text = "⚠️ Limited biodiversity — few key species detected."
    else:
        box_color = "#c62828"
        biodiversity_text = "🚫 No key species identified."

    # --- Ecosystem stability summary ---
    if diversity_score > 0.75 and evenness > 0.6:
        summary_text = "Stable ecosystem — balanced and resilient community structure."
    elif diversity_score > 0.4:
        summary_text = "Moderately stable ecosystem with partial species balance."
    else:
        summary_text = "Low stability — one or two species dominate the environment."

    # --- Unified Tier-Colored Display Box ---
    st.markdown(f"""
    <div style='
        background:rgba(0, 24, 61, 0.85);
        border-radius:15px;
        border:2px solid {box_color};
        box-shadow:0 8px 25px rgba(0,0,0,0.4);
        backdrop-filter:blur(12px);
        padding:1.8rem;
        margin-bottom:1.5rem;
        color:#E3F2FD;
        text-align:center;
    '>
        <h3 style='color:#64B5F6;'>Ecosystem Overview</h3>
        <p style='font-size:1.05rem; color:#90CAF9; margin-bottom:0.8rem;'><b>{status}</b></p>
        <hr style='border:0; height:1px; background:rgba(144,202,249,0.4); margin:1rem 0;'>
        <p style='font-size:0.95rem; line-height:1.6; color:#BBDEFB;'>
            <b>Shannon Index:</b> {diversity_score:.3f} • 
            <b>Evenness:</b> {evenness:.3f} • 
            <b>Indicator Species:</b> {keystone_count}/7<br>
            <b>Dominant Species:</b> {dominant_species} ({dominant_percent:.1f}%)
        </p>
        <p style='font-size:0.95rem; color:#B3E5FC; margin-top:1.2rem; line-height:1.6;'>
            {biodiversity_text}<br>{summary_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Charts ---
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        plot_biodiversity_pie(results_all)
    with col2:
        plot_species_abundance(results_all)
    st.markdown('</div>', unsafe_allow_html=True)