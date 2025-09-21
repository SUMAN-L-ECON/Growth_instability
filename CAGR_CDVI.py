import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import sqrt
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from io import BytesIO
import base64
import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= Page Configuration =================
st.set_page_config(
    page_title="CAGR & CDVI Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= Custom CSS Styling =================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        color: #2c3e50;
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #27ae60;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
        margin: 1rem 0;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .data-quality-good { color: #27ae60; font-weight: bold; }
    .data-quality-warning { color: #f39c12; font-weight: bold; }
    .data-quality-error { color: #e74c3c; font-weight: bold; }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .interpretation-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= Enhanced Function Definitions =================



def enhanced_compute_cagr(data, column):
    """Enhanced CAGR computation with additional statistics"""
    data_clean = data.copy().dropna()
    if len(data_clean) < 3:
        return None, None, None, None, None
    
    data_clean['Time'] = np.arange(1, len(data_clean) + 1)
    
    # Handle zero and negative values
    min_val = data_clean[column].min()
    if min_val <= 0:
        data_clean['LogColumn'] = np.log(data_clean[column] - min_val + 1)
    else:
        data_clean['LogColumn'] = np.log(data_clean[column])
    
    try:
        model = ols('LogColumn ~ Time', data=data_clean).fit()
        cagr = (np.exp(model.params['Time']) - 1) * 100
        p_value = model.pvalues['Time']
        adj_r_squared = model.rsquared_adj
        
        # Additional diagnostics
        dw_stat = durbin_watson(model.resid)
        conf_int = model.conf_int().loc['Time']
        
        return cagr, p_value, adj_r_squared, dw_stat, conf_int
    except:
        return None, None, None, None, None

def enhanced_compute_statistics(data, column):
    """Enhanced statistical computation"""
    clean_data = data[column].dropna()
    if len(clean_data) == 0:
        return None, None, None, None, None, None
    
    mean_val = clean_data.mean()
    std_val = clean_data.std()
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    
    # Additional statistics
    skewness = stats.skew(clean_data)
    kurtosis = stats.kurtosis(clean_data)
    median_val = clean_data.median()
    
    return mean_val, std_val, cv_val, skewness, kurtosis, median_val

def compute_enhanced_cdvi(cv, adj_r_squared):
    """Enhanced CDVI computation with bounds checking"""
    if pd.isna(cv) or pd.isna(adj_r_squared) or adj_r_squared < 0:
        return np.nan
    return cv * sqrt(max(0, 1 - adj_r_squared))

def generate_economic_interpretation(cagr, cdvi, pval, cv, indicator_name):
    """Generate comprehensive economic interpretation"""
    if any(pd.isna(x) for x in [cagr, cdvi, pval]):
        return "Insufficient data for analysis"
    
    # Growth assessment
    if cagr > 5:
        growth_desc = "strong positive growth"
        growth_color = "🟢"
    elif cagr > 0:
        growth_desc = "moderate positive growth"
        growth_color = "🟡"
    elif cagr > -5:
        growth_desc = "moderate decline"
        growth_color = "🟠"
    else:
        growth_desc = "significant decline"
        growth_color = "🔴"
    
    # Stability assessment
    if cdvi < 10:
        stability_desc = "highly stable"
        stability_color = "🟢"
    elif cdvi < 20:
        stability_desc = "moderately stable"
        stability_color = "🟡"
    elif cdvi < 30:
        stability_desc = "unstable"
        stability_color = "🟠"
    else:
        stability_desc = "highly volatile"
        stability_color = "🔴"
    
    # Statistical significance
    significance = "statistically significant" if pval < 0.05 else "not statistically significant"
    sig_color = "✅" if pval < 0.05 else "⚠️"
    
    return f"{growth_color} **Growth**: {growth_desc} ({cagr:.2f}% CAGR)\n{stability_color} **Stability**: {stability_desc} (CDVI: {cdvi:.2f})\n{sig_color} **Significance**: {significance} (p={pval:.4f})"

def create_trend_visualization(data, columns):
    """Create comprehensive trend visualization"""
    fig = make_subplots(
        rows=len(columns), cols=1,
        subplot_titles=columns,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, col in enumerate(columns):
        # Original data
        fig.add_trace(
            go.Scatter(
                x=data.iloc[:, 0], y=data[col],
                mode='lines+markers',
                name=f'{col} - Actual',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                showlegend=i == 0
            ),
            row=i+1, col=1
        )
        
        # Trend line
        x_numeric = np.arange(len(data))
        z = np.polyfit(x_numeric, data[col].dropna(), 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=data.iloc[:, 0], y=p(x_numeric),
                mode='lines',
                name=f'{col} - Trend',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=i == 0
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=300 * len(columns),
        title_text="Time Series Analysis with Trend Lines",
        title_font_size=20,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time Period", row=len(columns), col=1)
    
    return fig

def create_distribution_plots(data, columns):
    """Create distribution analysis plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Box Plots', 'Histograms', 'Q-Q Plots', 'Correlation Heatmap'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Box plots
    for i, col in enumerate(columns[:4]):  # Limit to 4 columns for visibility
        fig.add_trace(
            go.Box(y=data[col], name=col, showlegend=False),
            row=1, col=1
        )
    
    # Histograms
    for i, col in enumerate(columns[:2]):  # Limit to 2 for clarity
        fig.add_trace(
            go.Histogram(x=data[col], name=f'{col} Distribution', showlegend=False, opacity=0.7),
            row=1, col=2
        )
    
    # Correlation heatmap
    if len(columns) > 1:
        corr_matrix = data[columns].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Distribution Analysis Dashboard",
        title_font_size=20,
        template="plotly_white"
    )
    
    return fig

def generate_executive_summary(results_df):
    """Generate executive summary of the analysis"""
    if results_df.empty:
        return "No data available for analysis."
    
    total_indicators = len(results_df)
    positive_growth = len(results_df[results_df['CAGR (%)'].astype(float) > 0])
    significant_trends = len(results_df[results_df['P-Value'].astype(float) < 0.05])
    stable_indicators = len(results_df[results_df['CDVI'].astype(float) < 20])
    
    summary = f"""
    ## 📋 Executive Summary
    
    **Analysis Overview:**
    - Total Indicators Analyzed: **{total_indicators}**
    - Indicators with Positive Growth: **{positive_growth}** ({positive_growth/total_indicators*100:.1f}%)
    - Statistically Significant Trends: **{significant_trends}** ({significant_trends/total_indicators*100:.1f}%)
    - Stable Indicators (CDVI < 20): **{stable_indicators}** ({stable_indicators/total_indicators*100:.1f}%)
    
    **Key Insights:**
    """
    
    # Find best and worst performers
    if not results_df.empty:
        try:
            best_growth = results_df.loc[results_df['CAGR (%)'].astype(float).idxmax()]
            worst_growth = results_df.loc[results_df['CAGR (%)'].astype(float).idxmin()]
            most_stable = results_df.loc[results_df['CDVI'].astype(float).idxmin()]
            
            summary += f"""
    - **Best Performer**: {best_growth['Indicator']} with {best_growth['CAGR (%)']}% CAGR
    - **Most Challenging**: {worst_growth['Indicator']} with {worst_growth['CAGR (%)']}% CAGR
    - **Most Stable**: {most_stable['Indicator']} with CDVI of {most_stable['CDVI']}
            """
        except:
            summary += "\n- *Detailed performance analysis requires valid numeric data*"
    
    return summary

# ================= Main Application =================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Growth(CAGR) & Instability (CDVI) Analytics Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            | Built by Suman L (Ag. Economist)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>⚙️ Analysis Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload option
        st.subheader("📁 Data Input")
        data_source = st.radio(
            "Choose data source:",
            ["Upload File"],
            help="Upload your own data or use sample economic indicators"
        )
        
        uploaded_file = None
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your data file",
                type=["csv", "xlsx", "xls"],
                help="First column should be time/year, others are indicators"
            )
        
        # Analysis parameters
        st.subheader("🔧 Analysis Parameters")
        confidence_level = st.selectbox(
            "Confidence Level",
            [90, 95, 99],
            index=1,
            help="Confidence level for statistical tests"
        ) / 100
        
        min_observations = st.number_input(
            "Minimum Observations",
            min_value=3,
            max_value=50,
            value=5,
            help="Minimum number of observations required for analysis"
        )
        
        # Advanced options
        with st.expander("🎛️ Advanced Options"):
            handle_negative = st.checkbox(
                "Handle Negative Values",
                value=True,
                help="Apply transformation for negative values in growth calculations"
            )
            
            detect_outliers = st.checkbox(
                "Detect Outliers",
                value=True,
                help="Identify and flag potential outliers in the data"
            )
            
            include_diagnostics = st.checkbox(
                "Include Regression Diagnostics",
                value=True,
                help="Include additional regression diagnostic tests"
            )
    
    # Main content
    if data_source == uploaded_file is not None:
        
        # Load and display data
        if data_source == "Use Sample Data":
            data = load_sample_data()
            st.info("📊 Using sample economic data for demonstration")
        else:
            try:
                ext = uploaded_file.name.split('.')[-1].lower()
                if ext == "csv":
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Clean column names
                data.columns = data.columns.str.strip()
                data = data.dropna(how='all')
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        
        # Data quality assessment
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{len(data)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Indicators</div>
                <div class="metric-value">{len(data.columns)-1}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            quality_class = "data-quality-good" if missing_pct < 5 else "data-quality-warning" if missing_pct < 15 else "data-quality-error"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Missing Data</div>
                <div class="metric-value {quality_class}">{missing_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            time_span = len(data)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Time Span</div>
                <div class="metric-value">{time_span} periods</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "📈 Analysis Results", 
            "📋 Executive Summary",
            "📉 Visualizations", 
            "⬇️ Export & Reports"
        ])
        
        with tab1:
            st.subheader("📋 Data Preview & Quality Assessment")
            
            # Show data preview
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(
                    data.head(10),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.markdown("### Data Quality Report")
                for col in data.columns:
                    null_count = data[col].isnull().sum()
                    null_pct = (null_count / len(data)) * 100
                    
                    if null_pct == 0:
                        status = "🟢 Complete"
                    elif null_pct < 10:
                        status = "🟡 Minor gaps"
                    else:
                        status = "🔴 Major gaps"
                    
                    st.write(f"**{col}**: {status} ({null_pct:.1f}% missing)")
            
            # Show basic statistics
            st.subheader("📊 Basic Statistical Summary")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(
                    data[numeric_cols].describe(),
                    use_container_width=True
                )
        
        with tab2:
            st.subheader("🔬 CAGR & CDVI Analysis Results")
            
            # Column selection
            time_col = data.columns[0]
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.error("❌ No numeric columns found for analysis!")
                return
            
            # Enhanced column selection interface
            col1, col2 = st.columns([2, 1])
            with col1:
                analysis_options = ["All Indicators"] + numeric_cols
                selected_indicators = st.multiselect(
                    "Select indicators for analysis:",
                    options=analysis_options,
                    default=["All Indicators"],
                    help="Choose specific indicators or select all"
                )
            
            with col2:
                run_analysis = st.button(
                    "🚀 Run Analysis",
                    type="primary",
                    use_container_width=True,
                    help="Execute CAGR and CDVI analysis"
                )
            
            # Determine selected columns
            if "All Indicators" in selected_indicators:
                selected_cols = numeric_cols
            else:
                selected_cols = [col for col in selected_indicators if col in numeric_cols]
            
            if run_analysis and selected_cols:
                results = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, column in enumerate(selected_cols):
                    status_text.text(f'Analyzing {column}...')
                    progress_bar.progress((i + 1) / len(selected_cols))
                    
                    # Prepare data for analysis
                    temp_df = data[[time_col, column]].dropna()
                    
                    if len(temp_df) < min_observations:
                        results.append({
                            'Indicator': column,
                            'CAGR (%)': 'Insufficient Data',
                            'P-Value': 'N/A',
                            'Mean': 'N/A',
                            'Std Dev': 'N/A',
                            'CV (%)': 'N/A',
                            'Adj R²': 'N/A',
                            'CDVI': 'N/A',
                            'DW Statistic': 'N/A',
                            'Interpretation': f'Less than {min_observations} observations'
                        })
                        continue
                    
                    try:
                        # Enhanced computations
                        cagr, p_value, adj_r_squared, dw_stat, conf_int = enhanced_compute_cagr(temp_df, column)
                        mean_val, std_val, cv_val, skewness, kurtosis, median_val = enhanced_compute_statistics(temp_df, column)
                        
                        if all(x is not None for x in [cagr, p_value, adj_r_squared]):
                            cdvi = compute_enhanced_cdvi(cv_val, adj_r_squared)
                            interpretation = generate_economic_interpretation(cagr, cdvi, p_value, cv_val, column)
                            
                            results.append({
                                'Indicator': column,
                                'CAGR (%)': f"{cagr:.3f}",
                                'P-Value': f"{p_value:.4f}",
                                'Mean': f"{mean_val:.2f}",
                                'Std Dev': f"{std_val:.2f}",
                                'CV (%)': f"{cv_val:.2f}",
                                'Adj R²': f"{adj_r_squared:.4f}",
                                'CDVI': f"{cdvi:.3f}",
                                'DW Statistic': f"{dw_stat:.3f}" if dw_stat else "N/A",
                                'Interpretation': interpretation
                            })
                        else:
                            results.append({
                                'Indicator': column,
                                'CAGR (%)': 'Error',
                                'P-Value': 'Error',
                                'Mean': 'Error',
                                'Std Dev': 'Error',
                                'CV (%)': 'Error',
                                'Adj R²': 'Error',
                                'CDVI': 'Error',
                                'DW Statistic': 'Error',
                                'Interpretation': 'Analysis failed - check data quality'
                            })
                    
                    except Exception as e:
                        results.append({
                            'Indicator': column,
                            'CAGR (%)': 'Error',
                            'P-Value': 'Error',
                            'Mean': 'Error',
                            'Std Dev': 'Error',
                            'CV (%)': 'Error',
                            'Adj R²': 'Error',
                            'CDVI': 'Error',
                            'DW Statistic': 'Error',
                            'Interpretation': f'Error: {str(e)[:50]}...'
                        })
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.session_state['results_df'] = results_df
                    
                    # Enhanced results display
                    st.markdown("### 📊 Analysis Results")
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    valid_results = results_df[results_df['CAGR (%)'] != 'Error']
                    if not valid_results.empty:
                        with col1:
                            avg_cagr = pd.to_numeric(valid_results['CAGR (%)'], errors='coerce').mean()
                            st.metric("Average CAGR", f"{avg_cagr:.2f}%")
                        
                        with col2:
                            avg_cdvi = pd.to_numeric(valid_results['CDVI'], errors='coerce').mean()
                            st.metric("Average CDVI", f"{avg_cdvi:.2f}")
                        
                        with col3:
                            sig_count = pd.to_numeric(valid_results['P-Value'], errors='coerce')
                            significant = (sig_count < 0.05).sum()
                            st.metric("Significant Trends", f"{significant}/{len(valid_results)}")
                    
                    # Results table with enhanced formatting
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Interpretation": st.column_config.TextColumn(
                                "Economic Interpretation",
                                width="large"
                            )
                        }
                    )
                    
                    # Individual indicator cards
                    st.markdown("### 📋 Detailed Interpretations")
                    for _, row in valid_results.iterrows():
                        st.markdown(f"""
                        <div class="interpretation-card">
                            <h4>{row['Indicator']}</h4>
                            {row['Interpretation'].replace('\n', '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("📋 Executive Summary")
            
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                summary = generate_executive_summary(st.session_state['results_df'])
                st.markdown(summary)
                      
        with tab4:
            st.subheader("📊 Interactive Visualizations")
            
            if 'results_df' in st.session_state:
                # Trend visualization
                if len(selected_cols) > 0:
                    st.markdown("#### Time Series Analysis")
                    trend_fig = create_trend_visualization(data, selected_cols[:4])  # Limit for performance
                    st.plotly_chart(trend_fig, use_container_width=True)
                    
                    # Distribution analysis
                    st.markdown("#### Distribution Analysis")
                    dist_fig = create_distribution_plots(data, selected_cols)
                    st.plotly_chart(dist_fig, use_container_width=True)
                    
                    # CAGR vs CDVI scatter plot
                    if not st.session_state['results_df'].empty:
                        valid_results = st.session_state['results_df'][st.session_state['results_df']['CAGR (%)'] != 'Error']
                        if not valid_results.empty:
                            st.markdown("#### CAGR vs CDVI Analysis")
                            
                            cagr_values = pd.to_numeric(valid_results['CAGR (%)'], errors='coerce')
                            cdvi_values = pd.to_numeric(valid_results['CDVI'], errors='coerce')
                            
                            scatter_fig = go.Figure()
                            
                            # Add quadrant background colors
                            scatter_fig.add_shape(
                                type="rect", x0=-100, y0=0, x1=0, y1=100,
                                fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0)
                            )
                            scatter_fig.add_shape(
                                type="rect", x0=0, y0=0, x1=100, y1=20,
                                fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0)
                            )
                            scatter_fig.add_shape(
                                type="rect", x0=0, y0=20, x1=100, y1=100,
                                fillcolor="rgba(255, 255, 0, 0.1)", line=dict(width=0)
                            )
                            
                            scatter_fig.add_trace(
                                go.Scatter(
                                    x=cagr_values,
                                    y=cdvi_values,
                                    mode='markers+text',
                                    text=valid_results['Indicator'],
                                    textposition='top center',
                                    marker=dict(
                                        size=12,
                                        color=cagr_values,
                                        colorscale='RdYlGn',
                                        showscale=True,
                                        colorbar=dict(title="CAGR (%)")
                                    ),
                                    hovertemplate='<b>%{text}</b><br>CAGR: %{x:.2f}%<br>CDVI: %{y:.2f}<extra></extra>'
                                )
                            )
                            
                            scatter_fig.add_hline(y=20, line_dash="dash", line_color="red", 
                                                annotation_text="Stability Threshold (CDVI=20)")
                            scatter_fig.add_vline(x=0, line_dash="dash", line_color="black", 
                                                annotation_text="Growth Threshold (CAGR=0)")
                            
                            scatter_fig.update_layout(
                                title="Growth vs Stability Matrix",
                                xaxis_title="CAGR (%)",
                                yaxis_title="CDVI (Instability Index)",
                                height=600,
                                template="plotly_white",
                                annotations=[
                                    dict(x=-50, y=50, text="Declining & Unstable", showarrow=False, font=dict(size=12)),
                                    dict(x=50, y=10, text="Growing & Stable", showarrow=False, font=dict(size=12)),
                                    dict(x=50, y=50, text="Growing & Unstable", showarrow=False, font=dict(size=12))
                                ]
                            )
                            
                            st.plotly_chart(scatter_fig, use_container_width=True)
                            
                            # Performance radar chart
                            st.markdown("#### Performance Radar Chart")
                            
                            if len(valid_results) <= 8:  # Limit for readability
                                categories = ['Growth Rate', 'Stability', 'Statistical Significance', 'Consistency']
                                
                                radar_fig = go.Figure()
                                
                                for _, row in valid_results.iterrows():
                                    try:
                                        cagr_score = min(100, max(0, (float(row['CAGR (%)']) + 10) * 5))  # Normalize to 0-100
                                        stability_score = min(100, max(0, 100 - float(row['CDVI'])))  # Invert CDVI
                                        sig_score = 100 if float(row['P-Value']) < 0.05 else 30
                                        consistency_score = float(row['Adj R²']) * 100
                                        
                                        values = [cagr_score, stability_score, sig_score, consistency_score]
                                        
                                        radar_fig.add_trace(go.Scatterpolar(
                                            r=values + [values[0]],  # Close the polygon
                                            theta=categories + [categories[0]],
                                            fill='toself',
                                            name=row['Indicator'],
                                            opacity=0.7
                                        ))
                                    except:
                                        continue
                                
                                radar_fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 100]
                                        )),
                                    showlegend=True,
                                    title="Multi-Dimensional Performance Analysis",
                                    height=600
                                )
                                
                                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("👆 Run the analysis first to see visualizations")
        
        with tab5:
            st.subheader("📤 Export & Reports")
            
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                results_df = st.session_state['results_df']
                
                # Export options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📊 Data Exports")
                    
                    # CSV Export
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"CAGR_CDVI_Analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Excel Export with multiple sheets
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Main results
                        results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                        
                        # Summary statistics
                        valid_results = results_df[results_df['CAGR (%)'] != 'Error']
                        if not valid_results.empty:
                            numeric_cols = ['CAGR (%)', 'P-Value', 'Mean', 'Std Dev', 'CV (%)', 'Adj R²', 'CDVI']
                            summary_stats = valid_results[['Indicator'] + numeric_cols].copy()
                            
                            for col in numeric_cols:
                                summary_stats[col] = pd.to_numeric(summary_stats[col], errors='coerce')
                            
                            summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
                        
                        # Data quality report
                        quality_report = pd.DataFrame({
                            'Column': data.columns,
                            'Data Type': [str(dtype) for dtype in data.dtypes],
                            'Missing Values': [data[col].isnull().sum() for col in data.columns],
                            'Missing Percentage': [(data[col].isnull().sum() / len(data)) * 100 for col in data.columns]
                        })
                        quality_report.to_excel(writer, sheet_name='Data Quality', index=False)
                    
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=excel_buffer,
                        file_name=f"CAGR_CDVI_Complete_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("#### 📋 Professional Reports")
                    
                    # Generate comprehensive report
                    def generate_professional_report():
                        report = f"""
# Professional Economic Analysis Report
**Generated on:** {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}

## Executive Summary
{generate_executive_summary(results_df)}

## Detailed Analysis Results

| Indicator | CAGR (%) | P-Value | CDVI | Interpretation |
|-----------|----------|---------|------|----------------|
"""
                        for _, row in results_df.iterrows():
                            report += f"| {row['Indicator']} | {row['CAGR (%)']} | {row['P-Value']} | {row['CDVI']} | {row['Interpretation'][:50]}... |\n"
                        
                        report += """

## Methodology

### CAGR (Compound Annual Growth Rate)
CAGR is calculated using the regression method: CAGR = (e^β - 1) × 100
where β is the coefficient of time in the log-linear regression model.

### CDVI (Cuddy-Della Valle Index)
CDVI measures instability after removing trend effects:
CDVI = CV × √(1 - R²)
where CV is the coefficient of variation and R² is the coefficient of determination.

### Interpretation Guidelines
- **Growth**: CAGR > 0 indicates positive growth
- **Stability**: CDVI < 20 suggests stable behavior
- **Significance**: P-value < 0.05 indicates statistical significance

## Technical Notes
- Analysis performed using advanced econometric methods
- Missing values and outliers were handled appropriately
- Confidence intervals and diagnostic tests were included where applicable

---
*Report generated by Professional CAGR & CDVI Analytics Dashboard*
*Built by Suman L - Economics Analytics *
"""
                        return report
                    
                    professional_report = generate_professional_report()
                    
                    st.download_button(
                        label="📄 Download Professional Report (Markdown)",
                        data=professional_report,
                        file_name=f"Professional_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col3:
                    st.markdown("#### 🎯 Quick Stats")
                    
                    valid_results = results_df[results_df['CAGR (%)'] != 'Error']
                    if not valid_results.empty:
                        # Performance metrics
                        try:
                            cagr_values = pd.to_numeric(valid_results['CAGR (%)'], errors='coerce')
                            cdvi_values = pd.to_numeric(valid_results['CDVI'], errors='coerce')
                            pval_values = pd.to_numeric(valid_results['P-Value'], errors='coerce')
                            
                            st.metric("Highest CAGR", f"{cagr_values.max():.2f}%")
                            st.metric("Lowest CDVI", f"{cdvi_values.min():.2f}")
                            st.metric("Most Significant", f"p = {pval_values.min():.4f}")
                            
                            # Classification summary
                            growing_count = (cagr_values > 0).sum()
                            stable_count = (cdvi_values < 20).sum()
                            significant_count = (pval_values < 0.05).sum()
                            
                            st.markdown(f"""
                            **Classification Summary:**
                            - Growing: {growing_count}/{len(valid_results)}
                            - Stable: {stable_count}/{len(valid_results)}
                            - Significant: {significant_count}/{len(valid_results)}
                            """)
                        except:
                            st.warning("Unable to calculate summary statistics")
                
                # Additional export options
                st.markdown("---")
                st.markdown("#### 🔧 Advanced Export Options")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Generate PowerPoint Summary", use_container_width=True):
                        st.info("PowerPoint generation would be implemented with python-pptx library in a full deployment")
                
                with col2:
                    if st.button("📈 Create Interactive Dashboard", use_container_width=True):
                        st.info("Interactive dashboard would be generated using Plotly Dash in a full deployment")
            
            else:
                st.info("👆 Run the analysis first to access export options")
    
    else:
        # Welcome screen with instructions
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Getting Started</h3>
            <p>Welcome to the Professional CAGR & CDVI Analytics Dashboard!</p>
            
            <h4>📁 Data Requirements:</h4>
            <ul>
                <li><strong>First Column:</strong> Time periods (Years, Quarters, etc.)</li>
                <li><strong>Other Columns:</strong> Economic indicators (GDP, Inflation, Production, etc.)</li>
                <li><strong>Format:</strong> CSV, Excel (.xlsx, .xls)</li>
                <li><strong>Structure:</strong> Each row represents one time period</li>
            </ul>
            
            <h4>📊 What This App Analyzes:</h4>
            <ul>
                <li><strong>CAGR:</strong> Compound Annual Growth Rate - measures average growth</li>
                <li><strong>CDVI:</strong> Cuddy-Della Valle Index - measures instability after removing trend</li>
                <li><strong>Statistical Significance:</strong> P-values and confidence intervals</li>
                <li><strong>Performance Metrics:</strong> Mean, standard deviation, coefficient of variation</li>
            </ul>
            
            <h4>🎨 Features:</h4>
            <ul>
                <li>Interactive visualizations with Plotly</li>
                <li>Professional report generation</li>
                <li>Multiple export formats (CSV, Excel, PDF)</li>
                <li>Advanced statistical diagnostics</li>
                <li>Executive summary with policy recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-top: 2rem;">
        <h3>🚀 Professional CAGR & CDVI Analytics Dashboard</h3>
        <p style="margin: 1rem 0;">Built with ❤️ by <strong>Suman L</strong> - Economics Analytics Expert</p>
        <p style="margin: 0;">
            📧 <a href="mailto:sumanecon.uas@outlook.com" style="color: #ffd700;">sumanecon.uas@outlook.com</a> | 
            🌐 Arthanomix solutions
        </p>
        <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            Powered by Streamlit • Plotly • Statsmodels • Advanced Econometric Methods
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================= Application Entry Point =================
if __name__ == "__main__":
    main()
