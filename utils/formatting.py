"""Formatting utilities for Customer Churn Prediction & Retention Intelligence Dashboard."""

from typing import Union


def format_currency(value: Union[int, float], compact: bool = False) -> str:
    """Format numeric values as Indian Currency (INR).
    
    Examples:
        125000 -> ₹1,25,000 or ₹1.25 L
        12500000 -> ₹1.25 Cr
    """
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "₹0"
    
    val = float(value)
    abs_val = abs(val)
    prefix = "-" if val < 0 else ""
    
    if compact:
        if abs_val >= 10_000_000:
            return f"{prefix}₹{abs_val / 10_000_000:.2f} Cr"
        elif abs_val >= 100_000:
            return f"{prefix}₹{abs_val / 100_000:.2f} Lakh"
        elif abs_val >= 1_000:
            return f"{prefix}₹{abs_val / 1_000:.1f} K"
        else:
            return f"{prefix}₹{abs_val:.0f}"

    # Standard Indian number formatting with commas
    rounded = int(round(abs_val))
    s = str(rounded)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        # Group remaining digits in pairs of 2 from right to left
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        formatted = ",".join(reversed(groups)) + "," + last3

    return f"{prefix}₹{formatted}"


def format_percentage(value: Union[int, float], decimals: int = 1) -> str:
    """Format ratio (0.0 to 1.0 or 0 to 100) as percentage string."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "0.0%"
    val = float(value)
    # If passed as 0.187, multiply by 100
    if -1.0 <= val <= 1.0 and val != 0:
        val = val * 100.0
    return f"{val:.{decimals}f}%"


def format_number(value: Union[int, float]) -> str:
    """Format counts with standard commas."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "0"
    val = float(value)
    if val.is_integer():
        return f"{int(val):,}"
    return f"{val:,.2f}"


def get_risk_badge(risk_level: str) -> str:
    """Return colored HTML badge for a risk level."""
    risk_level = str(risk_level).strip().capitalize()
    colors = {
        "High": {"bg": "#FEE2E2", "text": "#991B1B", "border": "#F87171", "icon": "🔴"},
        "Medium": {"bg": "#FEF3C7", "text": "#92400E", "border": "#FBBF24", "icon": "🟡"},
        "Low": {"bg": "#D1FAE5", "text": "#065F46", "border": "#34D399", "icon": "🟢"},
    }
    cfg = colors.get(risk_level, {"bg": "#F1F5F9", "text": "#334155", "border": "#CBD5E1", "icon": "⚪"})
    return (
        f'<span style="background-color: {cfg["bg"]}; color: {cfg["text"]}; '
        f'border: 1px solid {cfg["border"]}; border-radius: 9999px; '
        f'padding: 3px 10px; font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">'
        f'{cfg["icon"]} {risk_level} Risk</span>'
    )


def get_priority_badge(priority: str) -> str:
    """Return colored HTML badge for customer retention priority."""
    p_str = str(priority).upper()
    if "P1" in p_str or "CRITICAL" in p_str:
        bg, text, border, icon = "#FEE2E2", "#7F1D1D", "#EF4444", "🚨"
        label = "P1 - Critical"
    elif "P2" in p_str or "HIGH" in p_str:
        bg, text, border, icon = "#FFEDD5", "#9A3412", "#FB923C", "⚡"
        label = "P2 - High"
    elif "P3" in p_str or "MEDIUM" in p_str:
        bg, text, border, icon = "#FEF3C7", "#92400E", "#FBBF24", "⚖️"
        label = "P3 - Medium"
    else:
        bg, text, border, icon = "#F1F5F9", "#475569", "#CBD5E1", "✓"
        label = "P4 - Low"

    return (
        f'<span style="background-color: {bg}; color: {text}; '
        f'border: 1px solid {border}; border-radius: 6px; '
        f'padding: 3px 8px; font-size: 0.8rem; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">'
        f'{icon} {label}</span>'
    )
