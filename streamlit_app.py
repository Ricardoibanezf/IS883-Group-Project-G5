import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from datetime import date

# Securely load the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI_API_KEY"]
client = OpenAI()

# Load the classification dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
try:
    df1 = pd.read_csv(url)
    st.write("Loaded CSV Data:")
    st.write(df1)
except Exception as e:
    st.error(f"Failed to load the dataset: {e}")

# Create input field for the user's complaint
client_complaint = st.text_area("Enter your complaint:", "Somebody stole money from my saving account")

# Check if complaint input is valid
if client_complaint:
    # Initialize classification process
    classified_data = []

    # Step 1: Classify by Product
    product_categories = df1['Product'].unique()
    response_product = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Product categories: {product_categories.tolist()}. "
                "Respond with the exact product as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_product = response_product.choices[0].message.content.strip()

    # Step 2: Classify by Sub-product
    subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].unique()
    response_subproduct = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Sub-product categories under the product '{assigned_product}': {subproduct_options.tolist()}. "
                "Respond with the exact sub-product as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_subproduct = response_subproduct.choices[0].message.content.strip()

    # Step 3: Classify by Issue
    issue_options = df1[(df1['Product'] == assigned_product) & 
                        (df1['Sub-product'] == assigned_subproduct)]['Issue'].unique()
    response_issue = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are a financial expert who classifies customer complaints based on these Issue categories under the product '{assigned_product}' and sub-product '{assigned_subproduct}': {issue_options.tolist()}. "
                "Respond with the exact issue as written there."
            )},
            {"role": "user", "content": f"This is my issue: '{client_complaint}'."}
        ],
        max_tokens=20,
        temperature=0.1
    )
    assigned_issue = response_issue.choices[0].message.content.strip()

    # Append results to classified_data
    classified_data.append({
        "Complaint": client_complaint,
        "Assigned Product": assigned_product,
        "Assigned Sub-product": assigned_subproduct,
        "Assigned Issue": assigned_issue
    })

  