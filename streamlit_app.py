import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd

# Title
st.title("💬 Financial Support Chatbot")

# Load the dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
st.write(f"Using dataset from: {url}")

try:
    df1 = pd.read_csv(url)
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")

# Extract unique product categories
product_categories = df1['Product'].unique().tolist()

# Initialize memory and the chatbot on the first run
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Initialize memory for the conversation
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=max_number_of_exchanges, return_messages=True
    )

    # Initialize the language model
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tool: Today's Date
    from langchain.agents import tool

    @tool
    def classify_complaint(complaint: str) -> str:
        """Classifies a complaint based on the dataset."""
        # Classify by Product
        product_match = None
        for product in product_categories:
            if product.lower() in complaint.lower():
                product_match = product
                break

        if not product_match:
            return "I'm sorry, I couldn't classify the complaint into a product category. Please provide more details."

        # Classify by Sub-product
        subproduct_options = df1[df1['Product'] == product_match]['Sub-product'].unique()
        subproduct_match = None
        for subproduct in subproduct_options:
            if subproduct.lower() in complaint.lower():
                subproduct_match = subproduct
                break

        if not subproduct_match:
            subproduct_match = "No specific sub-product match found."

        # Classify by Issue
        issue_options = df1[(df1['Product'] == product_match) &
                            (df1['Sub-product'] == subproduct_match)]['Issue'].unique()
        issue_match = None
        for issue in issue_options:
            if issue.lower() in complaint.lower():
                issue_match = issue
                break

        if not issue_match:
            issue_match = "No specific issue match found."

        # Format the classification response
        return (
            f"Complaint classified as:\n"
            f"- **Product:** {product_match}\n"
            f"- **Sub-product:** {subproduct_match}\n"
            f"- **Issue:** {issue_match}"
        )

    tools = [classify_complaint]

    # Prompt for complaint classification
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. "
                f"Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. "
                f"Use the tool to classify complaints accurately. Inform the user that a ticket has been created and provide the classification. "
                f"Reassure them that the issue will be forwarded to the appropriate support team. "
                f"Maintain a professional and empathetic tone throughout."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent with memory
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=st.session_state.memory, verbose=True
    )

# Display chat history
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Chat input
if user_input := st.chat_input("How can I help?"):
    st.chat_message("user").write(user_input)

    # Generate response from the agent
    response = st.session_state.agent_executor.invoke({"input": user_input})["output"]

    # Display the assistant's response
    st.chat_message("assistant").write(response)

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
    st.session_state.chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

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
            st.stop()
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
        chat = st.session_state.chat  # Ensure chat is referenced from session state
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
