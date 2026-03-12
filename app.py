import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="VoltSwap Battery Swapping Station Analysis",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Enhanced Styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main { background: radial-gradient(circle at top right, #1e293b, #0f172a); color: #f8fafc; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 { 
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    .stSidebar { background-color: #0f172a !important; border-right: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: white; border: none; border-radius: 12px; padding: 14px 28px;
        font-weight: 600; transition: all 0.3s ease;
    }
    .info-card {
        padding: 2rem; border-radius: 24px; background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 2rem; backdrop-filter: blur(12px);
    }
    </style>
""", unsafe_allow_html=True)

# Data Engine
@st.cache_data
def load_volt_data():
    np.random.seed(42)
    n_samples = 1000
    stations = [f'Station_{i}' for i in range(1, 6)]
    data = {
        'Station_ID': np.random.choice(stations, n_samples),
        'Time_of_Day': np.random.randint(0, 24, n_samples),
        'Battery_SoC': np.random.uniform(5, 30, n_samples),
        'Battery_Degradation': np.random.uniform(1, 5, n_samples),
        'Swaps_per_Hour': np.random.poisson(lam=8, size=n_samples),
        'Waiting_Time': np.random.exponential(scale=5, size=n_samples),
        'Operational_Cost': np.random.normal(loc=150, scale=15, size=n_samples)
    }
    df = pd.DataFrame(data)
    df['Revenue'] = df['Swaps_per_Hour'] * 200
    df['Net_Profit'] = df['Revenue'] - (df['Operational_Cost'] * df['Swaps_per_Hour'])
    df['Net_Profit'] = df['Net_Profit'].clip(lower=0)
    # Peak logic
    df.loc[(df['Time_of_Day'] >= 8) & (df['Time_of_Day'] <= 11), 'Swaps_per_Hour'] += 6
    df.loc[(df['Time_of_Day'] >= 17) & (df['Time_of_Day'] <= 21), 'Swaps_per_Hour'] += 8
    df['Revenue'] = df['Swaps_per_Hour'] * 200
    return df

# ML Logic
@st.cache_resource
def get_ml_assets(data_df):
    # Train Linear Regression
    X_reg = data_df[['Time_of_Day', 'Swaps_per_Hour', 'Waiting_Time']]
    y_reg = data_df['Revenue']
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)
    
    # Setup Scaler for Clustering (but don't mutate input data here)
    scaler = StandardScaler()
    scaler.fit(data_df[['Time_of_Day', 'Swaps_per_Hour', 'Waiting_Time']])
    
    return lr, scaler

# Initialize Data & Models
df_main = load_volt_data()
lr_model, feature_scaler = get_ml_assets(df_main)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>VoltSwap</h1>", unsafe_allow_html=True)
    st.write("---")
    nav = st.radio("Navigation", ["📊 EDA", "🧩 K-Means", "🔮 Linear Regression", "💰 Financial Feasibility"])
    st.write("---")
    st.info("VoltSwap Station Analytics Dashboard")

# --- EDA ---
if nav == "📊 EDA":
    st.title("VoltSwap Battery Swapping Station Analysis")
    st.subheader("EDA (Exploratory Data Analysis)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Swaps/Hr", f"{df_main['Swaps_per_Hour'].mean():.1f}")
    m2.metric("Avg Wait Time", f"{df_main['Waiting_Time'].mean():.1f}m")
    m3.metric("Total Revenue", f"₹{df_main['Revenue'].sum():,.0f}")
    m4.metric("Active Stations", "5")
    
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        trend = df_main.groupby('Time_of_Day')['Swaps_per_Hour'].mean().reset_index()
        fig = px.area(trend, x='Time_of_Day', y='Swaps_per_Hour', title="Hourly Demand Pattern", color_discrete_sequence=['#38bdf8'])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    with c2:
        st_rev = df_main.groupby('Station_ID')['Revenue'].mean().reset_index()
        fig2 = px.bar(st_rev, x='Station_ID', y='Revenue', title="Revenue per Station", color='Revenue', color_continuous_scale="Icefire")
        fig2.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, width='stretch')

# --- K-Means ---
elif nav == "🧩 K-Means":
    st.title("K-Means Clustering Analysis")
    k_val = st.sidebar.slider("Number of Clusters (K)", 2, 6, 3)
    
    # Process Clustering safely
    features = ['Time_of_Day', 'Swaps_per_Hour', 'Waiting_Time']
    X_scaled = feature_scaler.transform(df_main[features])
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    
    # Create local df for plotting to avoid cache mutation
    df_clu = df_main.copy()
    df_clu['Cluster'] = km.fit_predict(X_scaled).astype(str)
    
    fig3d = px.scatter_3d(df_clu, x='Time_of_Day', y='Swaps_per_Hour', z='Waiting_Time', 
                           color='Cluster', opacity=0.8, title=f"3D Demand Clusters (K={k_val})",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    fig3d.update_layout(template="plotly_dark", margin=dict(l=0,r=0,b=0,t=40))
    st.plotly_chart(fig3d, width='stretch')
    
    st.markdown("<div class='info-card'><h4>Operational Insights</h4><p>Clusters help identify <b>Peak Stress Points</b> (High Swaps + High Wait) and <b>Optimization Zones</b>.</p></div>", unsafe_allow_html=True)

# --- Linear Regression ---
elif nav == "🔮 Linear Regression":
    st.title("Linear Regression Revenue Forecasting")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("Predictive Inputs")
        h = st.slider("Hour", 0, 23, 14)
        s = st.slider("Swaps", 0, 40, 15)
        w = st.slider("Wait (Min)", 0, 45, 10)
        
        # Consistent feature name handling
        query = pd.DataFrame([[h, s, w]], columns=['Time_of_Day', 'Swaps_per_Hour', 'Waiting_Time'])
        pred = lr_model.predict(query)[0]
        
        st.markdown(f"""
        <div style='background: rgba(99, 102, 241, 0.2); padding: 30px; border-radius: 20px; border: 2px solid #6366f1; text-align: center;'>
            <h3 style='margin:0; opacity: 0.8;'>Predicted Revenue</h3>
            <h1 style='margin:10px 0; font-size: 3rem;'>₹{pred:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        sample = df_main.sample(200).copy()
        sample['Predicted'] = lr_model.predict(sample[['Time_of_Day', 'Swaps_per_Hour', 'Waiting_Time']])
        fig_r = px.scatter(sample, x='Revenue', y='Predicted', title="Model Performance (Actual vs Predicted)", opacity=0.6)
        # Add Reference Line
        m_val = max(sample['Revenue'].max(), sample['Predicted'].max())
        fig_r.add_trace(go.Scatter(x=[0, m_val], y=[0, m_val], mode='lines', name='Ideal Fit', line=dict(color='#818cf8', dash='dash')))
        fig_r.update_layout(template="plotly_dark")
        st.plotly_chart(fig_r, width='stretch')

# --- Financial ---
elif nav == "💰 Financial Feasibility":
    st.title("Financial Feasibility Analysis")
    st.subheader("Strategic Scenario Calculator")
    
    col_a, col_b = st.columns(2)
    u_inc = col_a.slider("Utilization Increase (%)", 0, 100, 60)
    c_red = col_b.slider("Cost Reduction (%)", 0, 50, 25)
    
    # Match Excel Logic exactly
    base_swaps = 500
    base_cost = 150
    stns = 5
    months = 36
    invest = 5000000
    
    new_swaps_mo = base_swaps * (1 + u_inc/100)
    new_op_cost = base_cost * (1 - c_red/100)
    profit_per_stn = (new_swaps_mo * 200) - (new_swaps_mo * new_op_cost)
    total_profit = profit_per_stn * stns * months
    net_ben = total_profit - invest
    
    res = st.columns(3)
    res[0].metric("Mo. Profit/Stn", f"₹{profit_per_stn:,.0f}")
    res[1].metric("Total 3-Yr Profit", f"₹{total_profit:,.0f}")
    res[2].metric("Net Benefit", f"₹{net_ben:,.0f}", delta=f"{((net_ben/invest)*100):.1f}%")
    
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=net_ben, title={'text': "3-Year Net Benefit (INR)"},
        gauge={'axis': {'range': [-invest, invest*3]}, 'bar': {'color': "#38bdf8"},
               'steps': [{'range': [-invest, 0], 'color': "#0f172a"}, {'range': [0, invest*3], 'color': "rgba(56, 189, 248, 0.1)"}]}
    ))
    gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(gauge, width='stretch')
    st.success("Project is financially viable with high ROI.")
