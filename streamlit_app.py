import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
import pandas as pd
from datetime import date

# Show title and description.
st.title("💬 Financial Support Chatbot")

# Add a text input field for the GitHub raw URL
url = st.text_input("Enter the GitHub raw URL of the CSV file:", 
                    "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv")

# Load the dataset if a valid URL is provided
if url:
    try:
        df1 = pd.read_csv(url)
        st.write("CSV Data:")
        st.write(df1)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Extract product, sub-product, and issue categories dynamically
if 'Product' in df1.columns:
    product_categories = df1['Product'].dropna().unique().tolist()
else:
    product_categories = []
    st.warning("The CSV file does not contain a 'Product' column.")

# Initialize session state
if "memory" not in st.session_state:  # IMPORTANT
    model_type = "gpt-4o-mini"

    # Initialize memory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        k=max_number_of_exchanges, 
        return_messages=True
    )

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tools
    from langchain.agents import tool

    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date. Use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date."""
        return "Today is " + str(date.today())

    tools = [datetoday]

    # Create the prompt
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. Wait for the user to describe their problem. Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. After assigning a product, identify the sub-product and issue categories using the CSV dataset. Respond with confirmation that the issue has been forwarded to the relevant team."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)


# Display existing chat messages
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Chat input and classification
if prompt := st.chat_input("How can I help?"):
    # Display user input
    st.chat_message("user").write(prompt)

    if df1 is not None and not df1.empty:
        try:
            # Step 1: Classify by Product
            response_product = st.session_state.agent_executor.invoke({"input": prompt})['output']
            assigned_product = next(
                (product for product in product_categories if product in response_product),
                None
            )

            if assigned_product:
                # Step 2: Classify by Sub-product
                subproduct_options = df1[df1['Product'] == assigned_product]['Sub-product'].dropna().unique().tolist()
                response_subproduct = st.session_state.agent_executor.invoke({"input": f"Classify this issue into a sub-product: {prompt}"})['output']
                assigned_subproduct = next(
                    (subproduct for subproduct in subproduct_options if subproduct in response_subproduct),
                    None
                )

                if assigned_subproduct:
                    # Step 3: Classify by Issue
                    issue_options = df1[(df1['Product'] == assigned_product) & 
                                        (df1['Sub-product'] == assigned_subproduct)]['Issue'].dropna().unique().tolist()
                    response_issue = st.session_state.agent_executor.invoke({"input": f"Classify this issue: {prompt}"})['output']
                    assigned_issue = next(
                        (issue for issue in issue_options if issue in response_issue),
                        None
                    )

                    if assigned_issue:
                        # Respond to user
                        st.session_state.memory.buffer.append({"type": "assistant", "content": (
                            f"Thank you for your message. Your issue has been classified under:\n"
                            f"- Product: {assigned_product}\n"
                            f"- Sub-product: {assigned_subproduct}\n"
                            f"- Issue: {assigned_issue}\n"
                            "A ticket has been created, and it has been forwarded to the appropriate team. They will reach out to you shortly."
                        )})
                    else:
                        st.session_state.memory.buffer.append({"type": "assistant", "content": "Could not classify the issue. Forwarding to a human assistant."})
                else:
                    st.session_state.memory.buffer.append({"type": "assistant", "content": "Could not classify the sub-product. Forwarding to a human assistant."})
            else:
                st.session_state.memory.buffer.append({"type": "assistant", "content": "Could not classify the product. Forwarding to a human assistant."})

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.session_state.memory.buffer.append({"type": "assistant", "content": "The complaint could not be classified due to missing or invalid data."})

    # Display response
    for message in st.session_state.memory.buffer:
        st.chat_message(message.get("type", "assistant")).write(message.get("content", ""))
