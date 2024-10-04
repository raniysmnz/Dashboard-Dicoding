# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Helper function to create daily orders DataFrame
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",   # Count unique orders
        "price": "sum"           # Sum the prices for revenue
    }).reset_index()
    
    # Rename the columns for clarity
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    
    return daily_orders_df

# Helper function to create summary of order items by product category
def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name_english").order_id.count().sort_values(ascending=False).reset_index()
    return sum_order_items_df

# Helper function to create state-wise customer count DataFrame
def create_bystate_df(df):
    bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    return bystate_df

# Helper function to create RFM (Recency, Frequency, Monetary) analysis DataFrame
def create_rfm_df(df, orders_df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", # Get the last order date (Recency)
        "order_id": "nunique",             # Count of orders (Frequency)
        "price": "sum"                     # Total price paid (Monetary)
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    # Ensure both timestamps are in datetime format
    rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"], errors='coerce')
    orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])

    # Calculate Recency as the number of days since the last order
    recent_date = orders_df["order_purchase_timestamp"].max()
    rfm_df["recency"] = (recent_date - rfm_df["max_order_timestamp"]).dt.days

    return rfm_df

# Load the data
all_df = pd.read_csv("all_data.csv")
orders_df = pd.read_csv("orders_dataset.csv")

# Ensure datetime columns are properly parsed
datetime_columns = ["order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date", "order_delivered_carrier_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column], errors='coerce')
    orders_df[column] = pd.to_datetime(orders_df[column], errors='coerce')

# Sort data by order purchase timestamp
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(drop=True, inplace=True)

# Sidebar filter for date range
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date, max_value=max_date, value=[min_date, max_date]
    )

# Filter data based on the selected date range
main_df = all_df[(all_df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) & 
                 (all_df["order_purchase_timestamp"] <= pd.to_datetime(end_date))]

# Create the necessary DataFrames for visualization
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
bystate_df = create_bystate_df(main_df)
rfm_df = create_rfm_df(main_df, orders_df)

# Header for Streamlit dashboard
st.header('E-Commerce Selling Dashboard :sparkles:')

# Daily orders visualization
st.subheader('Daily Orders')

# Show metrics for total orders and total revenue
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df['order_count'].sum()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df['revenue'].sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

# Plot daily orders
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(daily_orders_df["order_purchase_timestamp"], daily_orders_df["order_count"], marker='o', linewidth=2, color="#90CAF9")
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

# Best and worst performing products
st.subheader("Best & Worst Performing Product")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

# Best performing products
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(x="order_id", y="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_title("Best Performing Product", fontsize=15)
ax[0].tick_params(axis='y', labelsize=12)

# Worst performing products
sns.barplot(x="order_id", y="product_category_name_english", data=sum_order_items_df.tail(5), palette=colors, ax=ax[1])
ax[1].set_title("Worst Performing Product", fontsize=15)
ax[1].invert_xaxis()
ax[1].tick_params(axis='y', labelsize=12)

plt.suptitle("Best and Worst Performing Product by Number of Sales", fontsize=20)
st.pyplot(fig)

# Customer demographics
st.subheader("Customer Demographics")

fig, ax = plt.subplots(figsize=(20, 10))
colors = ["#90CAF9", "#D3D3D3"] * 4  # Adjusted to match color needs
sns.barplot(x="customer_count", y="customer_state", data=bystate_df.sort_values(by="customer_count", ascending=False), palette=colors, ax=ax)
ax.set_title("Number of Customers by States", fontsize=30)
st.pyplot(fig)

# RFM Analytics
st.subheader("Best Customer Based on RFM Parameters")

# Show metrics for RFM
col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df['recency'].mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df['frequency'].mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df['monetary'].mean(), "AUD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_monetary)

# Plot RFM parameters
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency").head(5), palette=colors, ax=ax[0])
ax[0].set_title("By Recency", fontsize=50)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_title("By Frequency", fontsize=50)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_title("By Monetary", fontsize=50)

st.pyplot(fig)

