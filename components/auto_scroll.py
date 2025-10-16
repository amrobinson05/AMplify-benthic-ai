import streamlit.components.v1 as components

def auto_scroll_to_results(anchor_id="results-anchor", flash_color="rgba(21,101,192,0.6)"):
    """
    Smoothly scrolls to the results section and flashes a highlight border.
    Works for both Classification and Detection pages.
    """
    components.html(f"""
    <script>
    (function(){{
        const doc = window.parent.document;
        const target = doc.querySelector("#{anchor_id}") || doc.querySelector('.results-box');
        if (!target) return;
        setTimeout(() => {{
            target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            target.style.transition = 'box-shadow 0.5s ease-in-out';
            target.style.boxShadow = '0 0 20px {flash_color}';
            setTimeout(() => {{ target.style.boxShadow = 'none'; }}, 900);
        }}, 100);
    }})();
    </script>
    """, height=0)
