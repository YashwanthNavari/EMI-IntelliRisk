import streamlit as st
from typing import Optional

def render_kpi_card(title: str, value: str, subtext: Optional[str] = None, delta_color: str = "normal"):
    """Render a modern glassmorphic KPI metric card."""
    subtext_html = f'<div class="kpi-subtext" style="color: {delta_color};">{subtext}</div>' if subtext else ""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_eligibility_badge(eligibility: str, confidence: Optional[float] = None):
    """Render a glowing status pill badge for EMI eligibility."""
    conf_text = f" ({confidence*100:.1f}% confidence)" if confidence is not None else ""
    if eligibility == "Eligible":
        st.markdown(f'<div class="badge-eligible">✓ ELIGIBLE{conf_text}</div>', unsafe_allow_html=True)
    elif eligibility == "High_Risk":
        st.markdown(f'<div class="badge-highrisk">⚠ HIGH RISK{conf_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-noteligible">✕ NOT ELIGIBLE{conf_text}</div>', unsafe_allow_html=True)

def render_factor_card(factor_type: str, title: str, detail: str):
    """Render an individual driving factor card."""
    cls_name = "factor-card-pos" if factor_type == "positive" else "factor-card-neg"
    icon = "✓" if factor_type == "positive" else "!"
    html = f"""
    <div class="{cls_name}">
        <div style="font-weight: 600; font-size: 0.95rem;">{icon} {title}</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">{detail}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
