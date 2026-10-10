%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime

# Page Configuration
st.set_page_config(
    page_title="Brim Gas Sales Prediction System",
    page_icon="⛽",
    layout="wide"
)

# Constants & Branding
UNIT_PRICE = 1500  # 1500 NGN per 1kg of gas
COMPANY_NAME = "Brim Gas Nig. Ltd, Jalingo, Taraba State"

st.sidebar.title("Navigation & Controls")
st.sidebar.markdown(f"**Company:** {COMPANY_NAME}")
st.sidebar.markdown(f"**Current Unit Price:** ₦{UNIT_PRICE:,} / kg")

menu = st.sidebar.radio("Select Module", [
    "📊 Dashboard & Overview", 
    "📂 Data Management & Upload", 
    "🤖 Model Training & Evaluation", 
    "📈 Sales Forecast & Prediction"
])

# Function to generate default sample historical data if none uploaded
@st.cache_data
def load_default_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-09-30", freq="D")
    n = len(dates)
    
    # Simulate realistic gas sales features
    df = pd.DataFrame({
        'Date': dates,
        'DayOfWeek': dates.dayofweek,
        'Month': dates.month,
        'IsWeekend': dates.dayofweek.isin([5, 6]).astype(int),
        'Inventory_Level_kg': np.random.randint(500, 5000, size=n),
        'Promotion_Active': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'Competitor_Price_kg': np.random.choice([1450, 1500, 1550], size=n),
    })
    
    # Target: Quantity Sold (kg) driven by features + noise
    base_sales = 300 + 50 * df['Promotion_Active'] - 0.02 * (df['Competitor_Price_kg'] - UNIT_PRICE)
    seasonal_effect = 50 * np.sin(2 * np.pi * df['Month'] / 12)
    df['Quantity_Sold_kg'] = np.clip(base_sales + seasonal_effect + np.random.normal(0, 40, size=n), 100, 1200).astype(int)
    df['Total_Revenue_NGN'] = df['Quantity_Sold_kg'] * UNIT_PRICE
    return df

# Initialize session state for data and model
if 'data' not in st.session_state:
    st.session_state['data'] = load_default_data()

if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['metrics'] = None

df = st.session_state['data']

# ================= MODULE 1: DASHBOARD & OVERVIEW =================
if menu == "📊 Dashboard & Overview":
    st.title("⛽ Brim Gas Nig. Ltd - Executive Sales Dashboard")
    st.markdown("Welcome to the **Sales Prediction and Decision Support System** designed for operational management in Jalingo, Taraba State[cite: 1].")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Records", f"{len(df):,} days")
    with col2:
        st.metric("Avg Daily Sales", f"{df['Quantity_Sold_kg'].mean():.1f} kg")
    with col3:
        st.metric("Total Revenue Generated", f"₦{df['Total_Revenue_NGN'].sum():,.2f}")
    with col4:
        st.metric("Fixed Unit Price", f"₦{UNIT_PRICE:,} / kg")
    
    st.markdown("---")
    st.subheader("Historical Sales Trend Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(df['Date']), df['Quantity_Sold_kg'], color='#1f77b4', linewidth=1)
    ax.set_title("Daily Gas Sales Volume (kg)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold (kg)")
    st.pyplot(fig)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Sales by Month (Seasonality)")
        monthly_sales = df.groupby('Month')['Quantity_Sold_kg'].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        monthly_sales.plot(kind='bar', color='#2ca02c', ax=ax2)
        ax2.set_ylabel("Avg Quantity (kg)")
        st.pyplot(fig2)
        
    with col_b:
        st.subheader("Impact of Promotions on Sales")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.boxplot(x='Promotion_Active', y='Quantity_Sold_kg', data=df, palette='Set2', ax=ax3)
        ax3.set_xticklabels(['No Promotion', 'Active Promotion'])
        st.pyplot(fig3)

# ================= MODULE 2: DATA MANAGEMENT & UPLOAD =================
elif menu == "📂 Data Management & Upload":
    st.title("📂 Data Sources & Management")
    st.markdown("Upload organizational sales records, inventory logs, and financial reports as specified in the system context[cite: 1].")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully uploaded!")
        st.session_state['data'] = user_df
        df = user_df
    
    st.subheader("Preview Dataset")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Summary Statistics")
    st.write(df.describe())

# ================= MODULE 3: MODEL TRAINING & EVALUATION =================
elif menu == "🤖 Model Training & Evaluation":
    st.title("🤖 Machine Learning Model Training")
    st.markdown("Train a Random Forest Regressor using historical sales logs, inventory status, and promotional factors.")
    
    if st.button("Train Model Now"):
        feature_cols = ['DayOfWeek', 'Month', 'IsWeekend', 'Inventory_Level_kg', 'Promotion_Active', 'Competitor_Price_kg']
        X = df[feature_cols]
        y = df['Quantity_Sold_kg']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        st.session_state['model'] = model
        st.session_state['metrics'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        st.success("Model trained successfully!")
        
    if st.session_state['metrics'] is not None:
        m = st.session_state['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error (MAE)", f"{m['MAE']:.2f} kg")
        col2.metric("Root Mean Squared Error (RMSE)", f"{m['RMSE']:.2f} kg")
        col3.metric("R² Score", f"{m['R2']:.4f}")
        
        st.subheader("Feature Importance")
        model = st.session_state['model']
        feature_cols = ['DayOfWeek', 'Month', 'IsWeekend', 'Inventory_Level_kg', 'Promotion_Active', 'Competitor_Price_kg']
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        importances.plot(kind='barh', color='orange', ax=ax)
        ax.set_title("Feature Importance in Sales Prediction")
        st.pyplot(fig)
    else:
        st.info("Click 'Train Model Now' above to initialize and evaluate the machine learning model.")

# ================= MODULE 4: SALES FORECAST & PREDICTION =================
elif menu == "📈 Sales Forecast & Prediction":
    st.title("📈 Operational Sales Forecasting & Decision Support")
    st.markdown("Input operating parameters to generate predictive sales insights for management decision-making[cite: 1]. Note: Unit price is fixed at **₦1,500 / kg**.")
    
    if st.session_state['model'] is None:
        st.warning("Please train the model first in the 'Model Training & Evaluation' module.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sel_date = st.date_input("Prediction Date", datetime.date.today())
            inventory_input = st.number_input("Expected Inventory Level (kg)", min_value=100, max_value=10000, value=2500)
            promo_input = st.selectbox("Promotion Status", [0, 1], format_func=lambda x: "Active Promotion (1)" if x == 1 else "No Promotion (0)")
        
        with col2:
            competitor_price = st.number_input("Competitor Unit Price (₦/kg)", min_value=1000, max_value=2500, value=1500)
            day_of_week = sel_date.weekday()
            month = sel_date.month
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
        if st.button("Generate Prediction"):
            model = st.session_state['model']
            input_data = pd.DataFrame([[day_of_week, month, is_weekend, inventory_input, promo_input, competitor_price]],
                                      columns=['DayOfWeek', 'Month', 'IsWeekend', 'Inventory_Level_kg', 'Promotion_Active', 'Competitor_Price_kg'])
            
            predicted_qty = model.predict(input_data)[0]
            predicted_revenue = predicted_qty * UNIT_PRICE
            
            st.markdown("---")
            st.subheader("Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Sales Volume", f"{predicted_qty:,.2f} kg")
            res_col2.metric("Estimated Revenue (@ ₦1,500/kg)", f"₦{predicted_revenue:,.2f}")
            res_col3.metric("Unit Price Applied", f"₦{UNIT_PRICE:,} / kg")
