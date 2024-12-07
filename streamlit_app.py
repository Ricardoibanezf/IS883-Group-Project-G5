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

product_categories = df1['Product'].unique().tolist()

# Initialize chatbot memory and setup on first run
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Memory for the chatbot
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=10, return_messages=True
    )

    # Initialize the OpenAI chatbot
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tool: Today's Date
    from langchain.agents import tool

    @tool
    def classify_complaint(complaint: str) -> str:
        """Classifies a complaint based on the dataset."""
        product_match = next((p for p in product_categories if p.lower() in complaint.lower()), None)
        if not product_match:
            return "I'm sorry, I couldn't classify the complaint. Please provide more details."

        subproduct_options = df1[df1['Product'] == product_match]['Sub-product'].unique()
        subproduct_match = next((s for s in subproduct_options if s.lower() in complaint.lower()), "No match")

        issue_options = df1[
            (df1['Product'] == product_match) & (df1['Sub-product'] == subproduct_match)
        ]['Issue'].unique()
        issue_match = next((i for i in issue_options if i.lower() in complaint.lower()), "No match")

        return f"{product_match}|{subproduct_match}|{issue_match}"

    tools = [classify_complaint]

    # Chat prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a financial support assistant. Greet the user and classify complaints based on these categories: {product_categories}. "
                "Use the classify_complaint tool to determine the product, sub-product, and issue, then send the issue to Jira for ticket creation."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create chatbot agent
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=st.session_state.memory, verbose=True
    )

# Display chat history
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# User input
if user_input := st.chat_input("How can I help?"):
    st.chat_message("user").write(user_input)

    # Process user input
    response = st.session_state.agent_executor.invoke({"input": user_input})["output"]

    # Parse classification output
    if "|" in response:
        product, subproduct, assigned_issue = response.split("|")
        st.chat_message("assistant").write(
            f"Complaint classified as:\n"
            f"- Product: {product}\n"
            f"- Sub-product: {subproduct}\n"
            f"- Issue: {assigned_issue}\n\n"
            f"Creating a Jira task..."
        )

        # Jira integration
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
            f"The summary is '{assigned_issue}'. "
            f"Set the priority to 'Highest' if the issue involves fraud, otherwise set it to 'High'. "
            f"The description is '{user_input}'."
        )

        # Create the Jira agent executor
        chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")
        agent = create_react_agent(chat, tools, hub.pull("hwchase17/react"))
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Execute the task creation
        try:
            jira_result = agent_executor.invoke({"input": question})
            st.chat_message("assistant").write("Task successfully created in Jira!")
            st.json(jira_result)
        except Exception as e:
            st.chat_message("assistant").write(f"Failed to create the Jira task: {e}")
    else:
        st.chat_message("assistant").write(response)
