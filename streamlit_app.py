import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Streamlit App
st.title("Customer Complaint Chatbot")

# Load API Key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# Load Dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/refs/heads/main/Classification_data.csv"
df1 = pd.read_csv(url)

# Chatbot Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help classify your complaint. Please describe your issue."}
    ]

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# User Input
user_input = st.text_input("Your message:", key="user_input")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Classification process
    client_complaint = user_input

    # Classify by Product
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

    # Classify by Sub-product
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

    # Classify by Issue
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

    # Assistant response
    bot_response = (
        f"I understand your issue. Here is the classification:\n\n"
        f"- **Product:** {assigned_product}\n"
        f"- **Sub-product:** {assigned_subproduct}\n"
        f"- **Issue:** {assigned_issue}\n\n"
        f"If you have any further questions, let me know!"
    )
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Clear input box
    st.experimental_rerun()
