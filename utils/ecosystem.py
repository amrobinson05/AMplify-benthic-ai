import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter

plt.rcParams.update({
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "axes.edgecolor": "#64B5F6",
    "axes.labelcolor": "#E3F2FD",
    "xtick.color": "#E3F2FD",
    "ytick.color": "#E3F2FD",
    "text.color": "#E3F2FD",
    "font.weight": "bold",
    "axes.titleweight": "bold",
})


def calculate_ecosystem_health(results):
    """Compute adjusted ecosystem health score with richness penalty."""
    if not results:
        return 0.0, "No Data"

    species_list = [r.get("species") for r in results if r.get("species")]
    species_counts = Counter(species_list)
    total = sum(species_counts.values())
    S = len(species_counts)
    if total == 0 or S == 0:
        return 0.0, "No Data"

    # Shannon diversity (normalized)
    H = -sum((count/total) * math.log(count/total) for count in species_counts.values())
    max_H = math.log(S)
    normalized_H = H / max_H if max_H > 0 else 0

    richness_modifier = min(S / 7.0, 1.0)
    diversity_score = normalized_H * richness_modifier

    if diversity_score > 0.75:
        status = "🌿 Stable and Resilient Ecosystem"
    elif diversity_score > 0.4:
        status = "🪸 Functioning but Vulnerable Ecosystem"
    else:
        status = "⚠️ At-Risk Ecosystem"

    return diversity_score, status


def plot_biodiversity_pie(results):
    """Pie chart of species proportion with soft ocean glow."""
    species_list = [r.get("species") for r in results if r.get("species")]
    counts = Counter(species_list)
    if not counts:
        st.info("No data available for biodiversity chart.")
        return

    colors = [
        "#5fa8f3", "#90caf9", "#42a5f5",
        "#64b5f6", "#1e88e5", "#0d47a1", "#1565c0"
    ][:len(counts)]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        counts.values(),
        labels=counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax.set_title("Biodiversity Composition", fontsize=14, weight="bold", pad=20)
    fig.patch.set_alpha(0.0)
    st.pyplot(fig, transparent=True)


def plot_species_abundance(results):
    """Bar chart of species frequencies with glowing blue bars."""
    species_list = [r.get("species") for r in results if r.get("species")]
    counts = Counter(species_list)
    if not counts:
        st.info("No data available for abundance chart.")
        return

    species = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(species)))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(species, values, color=colors, edgecolor="#E3F2FD", linewidth=1.5)
    ax.set_title("Species Abundance", fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=12, weight="bold")

    for bar in bars:
        bar.set_alpha(0.9)
        bar.set_linewidth(1.2)
        bar.set_edgecolor("white")

    ax.grid(alpha=0.3, linestyle="--")
    plt.xticks(rotation=25, ha="right")
    fig.patch.set_alpha(0.0)
    st.pyplot(fig, transparent=True)
