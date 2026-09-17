"""
visualization.py
Intelligent chart generation matching the dark futuristic theme.
Supports Auto, Bar, Line, and Pie charts based on query DataFrame columns.
"""

from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def detect_chart_suitability(df: Optional[pd.DataFrame]) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Analyzes DataFrame to determine if it is suitable for visualization.
    Returns: (is_suitable, recommended_type, x_col, y_col)
    """
    if df is None or df.empty or len(df.columns) < 2:
        return False, "None", None, None
        
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "year", "month"])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    
    if not num_cols:
        return False, "None", None, None
        
    y_col = num_cols[-1]
    
    # Check for time-series / date column
    if date_cols and len(date_cols) > 0:
        return True, "Line", date_cols[0], y_col
        
    # Check for categorical + numerical
    if cat_cols and len(cat_cols) > 0:
        x_col = cat_cols[0]
        # If very few categories (3-7), pie is also viable
        if 2 <= len(df) <= 7:
            return True, "Bar", x_col, y_col
        return True, "Bar", x_col, y_col
        
    # Default: first column as label, second as value
    return True, "Bar", df.columns[0], y_col

def create_chart(
    df: Optional[pd.DataFrame], 
    chart_type: str = "Auto", 
    title: str = "Query Result Visualization"
) -> Optional[plt.Figure]:
    """
    Generates a high-quality modern dark-themed Matplotlib Figure.
    """
    is_suitable, rec_type, x_col, y_col = detect_chart_suitability(df)
    if not is_suitable or x_col is None or y_col is None:
        return None
        
    effective_type = rec_type if chart_type == "Auto" else chart_type
    
    # Modern dark palette
    bg_color = "#0f172a"      # Slate 900
    card_color = "#1e293b"    # Slate 800
    text_color = "#f8fafc"    # Slate 50
    grid_color = "#334155"    # Slate 700
    accent_colors = ["#8b5cf6", "#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#a855f7"]
    
    # Limit rows to 12 for clean aesthetics
    plot_df = df.head(12).copy()
    labels = plot_df[x_col].astype(str)
    values = pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0)
    
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(card_color)
    
    ax.tick_params(colors=text_color, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(0.8)
        
    ax.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    
    if effective_type == "Pie":
        ax.axis("off")
        colors = (accent_colors * 3)[:len(labels)]
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=labels, 
            autopct="%1.1f%%", 
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": bg_color, "linewidth": 1.5, "antialiased": True}
        )
        for t in texts:
            t.set_color(text_color)
            t.set_fontsize(9)
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(9)
            at.set_weight("bold")
            
    elif effective_type == "Line":
        ax.plot(labels, values, marker="o", color="#06b6d4", linewidth=2.5, markersize=6, label=y_col)
        ax.fill_between(labels, values, color="#06b6d4", alpha=0.15)
        ax.set_xlabel(x_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(y_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        plt.xticks(rotation=35, ha="right")
        
    elif effective_type == "Bar (Horizontal)" or (effective_type == "Bar" and len(labels) > 6 and max([len(str(l)) for l in labels]) > 10):
        # Horizontal bar if labels are long
        bars = ax.barh(labels, values, color="#8b5cf6", edgecolor="#a855f7", height=0.6)
        ax.set_xlabel(y_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(x_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        ax.invert_yaxis()
        
    else:  # Standard Vertical Bar
        bars = ax.bar(labels, values, color="#8b5cf6", edgecolor="#a855f7", width=0.55)
        ax.set_xlabel(x_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(y_col, color=text_color, fontsize=10, fontweight="bold", labelpad=8)
        plt.xticks(rotation=35, ha="right")
        
    ax.set_title(title, color=text_color, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
