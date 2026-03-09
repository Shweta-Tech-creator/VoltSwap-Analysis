import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Page Configuration - MUST BE AT THE VERY TOP
st.set_page_config(page_title="VoltSwap Station Analytics", layout="wide", page_icon="⚡")

# Custom CSS for better UI and Premium look
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Cache data generation
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    stations = [f'Station_{i}' for i in range(1, 6)]
    
    data = {
        'Station_ID': np.random.choice(stations, n_samples),
        'Time_of_Day': np.random.randint(0, 24, n_samples),
        'Battery_SoC_Initial': np.random.uniform(5, 30, n_samples),
        'Swaps_Per_Hour': np.random.poisson(lam=5, size=n_samples),
        'Wait_Time_Mins': np.random.exponential(scale=5, size=n_samples),
        'Operational_Cost_INR': np.random.normal(loc=150, scale=20, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    df['Revenue_INR'] = df['Swaps_Per_Hour'] * 200
    df['Net_Profit'] = df['Revenue_INR'] - df['Operational_Cost_INR'] * df['Swaps_Per_Hour']
    df['Net_Profit'] = df['Net_Profit'].apply(lambda x: max(0, x))
    return df

df = load_data()

# ML Model
@st.cache_resource
def train_model(data):
    X = data[['Time_of_Day', 'Swaps_Per_Hour', 'Wait_Time_Mins']]
    y = data['Revenue_INR']
    model = LinearRegression()
    # To avoid warning, supply feature names appropriately or fit as is
    model.fit(X.values, y.values)
    return model

model = train_model(df)

# Sidebar UI
with st.sidebar:
    st.title("⚡ VoltSwap Analytics")
    nav = st.radio("Navigation Menu", ["📊 Network Overview", "🤖 ML Revenue Predictor", "💼 Business Case Calculator"])
    st.markdown("---")
    st.info("**Case Study 52**\n\nEV Battery Swapping Analysis Project.", icon="ℹ️")

if nav == "📊 Network Overview":
    st.title("Network Performance Overview")
    st.markdown("Monitor real-time simulated statistics for all 5 smart battery swapping stations.")
    
    # Key Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Swaps Today", f"{df['Swaps_Per_Hour'].sum():,}", "+12% from yesterday")
    col2.metric("Average Wait Time", f"{df['Wait_Time_Mins'].mean():.1f} mins", "-1.5 mins")
    col3.metric("Total Revenue", f"₹{df['Revenue_INR'].sum():,.0f}", "+8.4%")
    col4.metric("Active Stations", "5 / 5", "100% Uptime")
    
    st.markdown("---")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Demand by Time of Day")
        hourly_demand = df.groupby('Time_of_Day')['Swaps_Per_Hour'].mean().reset_index()
        fig1 = px.area(hourly_demand, x='Time_of_Day', y='Swaps_Per_Hour', 
                       labels={'Time_of_Day': 'Hour of Day (0-23)', 'Swaps_Per_Hour': 'Average Swaps / Hour'},
                       color_discrete_sequence=['#38bdf8'], title="Average Hover Demand")
        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Station Revenue Comparison")
        fig2 = px.box(df, x='Station_ID', y='Revenue_INR', 
                      color='Station_ID',
                      labels={'Station_ID': 'Station Location', 'Revenue_INR': 'Revenue Generated (₹)'}, title="Revenue Distribution")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

elif nav == "🤖 ML Revenue Predictor":
    st.title("Hourly Revenue Prediction Model")
    st.markdown("Leverage our trained **Linear Regression** model to forecast revenue based on live conditions.")
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("### Input Live Conditions")
        in_time = st.slider("Hour of the Day (0-23)", 0, 23, 14)
        in_swaps = st.slider("Estimated Swaps/Hour", 1, 30, 8)
        in_wait = st.slider("Queue Wait Time (Mins)", 0, 45, 10)
        
    with c2:
        st.markdown("### AI Forecast Result")
        prediction = model.predict([[in_time, in_swaps, in_wait]])[0]
        
        st.info("Model Status: Active & Synced", icon="✅")
        st.success(f"## Predicted Revenue: ₹{prediction:,.2f}")
        
        # Simple gauge/progress visualization
        target = 3000
        progress_val = min(prediction / target, 1.0)
        st.progress(float(progress_val))
        st.caption(f"Progress towards peak hourly target of ₹{target:,.0f}")

elif nav == "💼 Business Case Calculator":
    st.title("Strategic Investment Calculator")
    st.markdown("Use this calculator to answer **Questions A1 & A2** of the case study. Adjust sliders to simulate Net Benefit over a 3-year term.")
    st.markdown("---")
    
    cc1, cc2 = st.columns(2)
    with cc1:
        inc_utilization = st.slider("Utilization Increase (%)", 0, 100, 60, step=5)
    with cc2:
        dec_opcost = st.slider("Operational Cost Reduction (%)", 0, 50, 25, step=5)
        
    # Baseline Constants
    stns = 5
    months = 36
    investments = 5000000
    base_swaps_mo = 500
    base_rev_per_swap = 200
    base_opcost = 150
    
    # Calculations
    new_swaps_mo = base_swaps_mo * (1 + (inc_utilization / 100))
    new_opcost = base_opcost * (1 - (dec_opcost / 100))
    monthly_rev_stn = new_swaps_mo * base_rev_per_swap
    total_rev_3years = monthly_rev_stn * stns * months
    net_benefit = total_rev_3years - investments
    
    st.markdown("### Financial Projection Summary")
    
    metrics = st.columns(3)
    metrics[0].metric("Monthly Revenue/Station", f"₹{monthly_rev_stn:,.0f}")
    metrics[1].metric("Total Initial Investment", f"₹{investments:,.0f}")
    metrics[2].metric("Estimated 3-Year Gross", f"₹{total_rev_3years:,.0f}")
    
    if net_benefit >= 0:
        st.success(f"# ✅ Projected Net Benefit: +₹{net_benefit:,.0f}")
        st.toast("Project is highly viable!", icon="🚀")
    else:
        st.error(f"# ❌ Projected Net Benefit: ₹{net_benefit:,.0f}")
