import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="VoltSwap Station Analysis", layout="wide", page_icon="⚡")

st.title("⚡ VoltSwap Pvt. Ltd. - Battery Swapping Station Dashboard")
st.markdown("Analyze efficiency, predict revenue, and calculate Net Benefit for the EV battery swapping network.")

# Cache the data generation
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
    df['Revenue_INR'] = df['Swaps_Per_Hour'] * 200  # Assume ₹200 per swap
    df['Net_Profit'] = df['Revenue_INR'] - df['Operational_Cost_INR'] * df['Swaps_Per_Hour']
    df['Net_Profit'] = df['Net_Profit'].apply(lambda x: max(0, x))
    return df

df = load_data()

# Train simple ML model
@st.cache_resource
def train_model(data):
    X = data[['Time_of_Day', 'Swaps_Per_Hour', 'Wait_Time_Mins']]
    y = data['Revenue_INR']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(df)

# Sidebar Navigation
nav = st.sidebar.radio("Navigation", ["Dashboard & EDA", "Interactive Predictions & Business Scenario"])

if nav == "Dashboard & EDA":
    st.header("📊 Exploratory Data Analysis (Synthetic Dataset)")
    st.write(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demand by Time of Day")
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Time_of_Day', y='Swaps_Per_Hour', estimator=np.mean, ax=ax, color="purple")
        ax.set_title("Average Battery Swaps by Hour")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Revenue Distribution by Station")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='Station_ID', y='Revenue_INR', ax=ax2, palette="viridis")
        ax2.set_title("Revenue generation across 5 simulated stations")
        st.pyplot(fig2)

    st.subheader("Station Key Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Total Swaps Generated", value=f"{df['Swaps_Per_Hour'].sum():,}")
    m2.metric(label="Total Revenue", value=f"₹{df['Revenue_INR'].sum():,.2f}")
    m3.metric(label="Average Wait Time", value=f"{df['Wait_Time_Mins'].mean():.1f} mins")

elif nav == "Interactive Predictions & Business Scenario":
    st.header("📈 Predictive Revenue Interface (Linear Regression)")
    st.markdown("Use the slider below to predict hourly station revenue based on active conditions.")
    
    col_input, col_pred = st.columns([1, 1])
    with col_input:
        in_time = st.slider("Time of Day (Hour 0-23)", 0, 23, 12)
        in_swaps = st.slider("Estimated Swaps Per Hour", 1, 25, 5)
        in_wait = st.slider("Current Wait Time (Minutes)", 0, 30, 5)
        
    with col_pred:
        prediction = model.predict([[in_time, in_swaps, in_wait]])[0]
        st.info("### AI Prediction Results Model Active")
        st.success(f"### Predicted Hourly Revenue: ₹{prediction:,.2f}")
        st.write("This prediction uses an underlying linear regression model trained on our dataset.")
        
    st.markdown("---")
    st.header("💰 Business Scenario Calculator: Net Benefit")
    st.markdown("Estimate Net Benefit with custom assumed utilizations (Case Study Q1 & Q2).")
    
    inc_utilization = st.number_input("Station Utilization Increase (%)", value=60)
    dec_opcost = st.number_input("Operational Cost Reduction (%)", value=25)
    
    # Base Math (Fixed single station baseline per month):
    base_swaps_mo = 500
    base_rev_per_swap = 200
    base_opcost = 150
    stns = 5
    months = 36
    investments = 5000000
    
    new_swaps_mo = base_swaps_mo * (1 + (inc_utilization / 100))
    new_opcost = base_opcost * (1 - (dec_opcost / 100))
    monthly_rev_stn = new_swaps_mo * base_rev_per_swap
    
    total_rev_3years = monthly_rev_stn * stns * months
    net_benefit = total_rev_3years - investments
    
    st.write(f"**Under these parameters:**")
    st.write(f"- Monthly Revenue per Station: ₹{monthly_rev_stn:,.2f}")
    st.write(f"- Total Project Revenue (3 years, 5 stations): ₹{total_rev_3years:,.2f}")
    st.write(f"- Initial Investment: ₹{investments:,.2f}")
    
    if net_benefit >= 0:
        st.success(f"### Calculated Net Benefit: +₹{net_benefit:,.2f} (Viable Project)")
    else:
        st.error(f"### Calculated Net Benefit: ₹{net_benefit:,.2f} (Not Viable)")

st.sidebar.markdown("---")
st.sidebar.info("Case Study 52: EV Battery Swapping Analysis")
