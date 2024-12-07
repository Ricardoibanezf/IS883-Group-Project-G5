import os
import streamlit as st
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd

# Title
st.title("💬 Financial Support Chatbot with Jira Integration")

# Load dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
st.write(f"Using dataset from: {url}")

try:
    df1 = pd.read_csv(url)
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()

product_categories = df1['Product'].unique().tolist()

# Initialize chatbot memory and setup on first run
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Memory for the chatbot
    st.session_state.memory = []
    st.session_state.needs_clarification = False

    # Initialize the OpenAI chatbot
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

# Display chat history
for message in st.session_state.memory:
    st.chat_message(message["role"]).write(message["content"])

# User input
if user_input := st.chat_input("How can I help?"):
    st.chat_message("user").write(user_input)
    st.session_state.memory.append({"role": "user", "content": user_input})

    # Handle clarification logic
    if st.session_state.needs_clarification:
        combined_input = f"{st.session_state.memory[-2]['content']} {user_input}"
        st.session_state.needs_clarification = False
    else:
        combined_input = user_input

    # Classification logic
    product_match = next((p for p in product_categories if p.lower() in combined_input.lower()), None)

    if not product_match:
        if not st.session_state.needs_clarification:
            st.session_state.needs_clarification = True
            bot_response = (
                "Could you please provide more details about your issue to help with classification?"
            )
            st.chat_message("assistant").write(bot_response)
            st.session_state.memory.append({"role": "assistant", "content": bot_response})
        else:
            product_match = "Miscellaneous"

    # Continue classification
    subproduct_options = df1[df1['Product'] == product_match]['Sub-product'].unique()
    subproduct_match = next((s for s in subproduct_options if s.lower() in combined_input.lower()), "General")

    issue_options = df1[
        (df1['Product'] == product_match) & (df1['Sub-product'] == subproduct_match)
    ]['Issue'].unique()
    issue_match = next((i for i in issue_options if i.lower() in combined_input.lower()), "General Issue")

    # Provide classification
    classification_response = (
        f"Complaint classified as:\n"
        f"- Product: {product_match}\n"
        f"- Sub-product: {subproduct_match}\n"
        f"- Issue: {issue_match}\n\n"
        f"Creating a Jira task..."
    )
    st.chat_message("assistant").write(classification_response)
    st.session_state.memory.append({"role": "assistant", "content": classification_response})

    # Jira task creation
    os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
    os.environ["JIRA_USERNAME"] = "rich@bu.edu"
    os.environ["JIRA_INSTANCE_URL"] = "https://is883-genai-r.atlassian.net/"
    os.environ["JIRA_CLOUD"] = "True"

    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    tools = toolkit.get_tools()

    # Fix tool names and descriptions
    for idx, tool in enumerate(tools):
        tools[idx].name = tools[idx].name.replace(" ", "_")
        if "create_issue" in tools[idx].name:
            tools[idx].description += " Ensure to specify the project ID."

    # Prepare task creation prompt
    question = (
        f"Create a task in my project with the key FST. Assign it to rich@bu.edu. "
        f"The summary is '{issue_match}'. "
        f"Set the priority to 'Highest' if the issue involves fraud, otherwise set it to 'High'. "
        f"The description is '{combined_input}'."
    )

    # Create the Jira agent executor
    try:
        agent = create_tool_calling_agent(chat, tools, ChatPromptTemplate.from_messages([]))
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        jira_result = agent_executor.invoke({"input": question})
        task_creation_response = "Task successfully created in Jira!"
        st.chat_message("assistant").write(task_creation_response)
        st.session_state.memory.append({"role": "assistant", "content": task_creation_response})
        st.json(jira_result)
    except Exception as e:
        error_response = f"Failed to create the Jira task: {e}"
        st.chat_message("assistant").write(error_response)
        st.session_state.memory.append({"role": "assistant", "content": error_response})
