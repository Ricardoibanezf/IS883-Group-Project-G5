import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub
import pandas as pd

# Show title and description.
st.title("💬 Financial Support Chatbot")

### Adding subproducts


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

st.write(df1)

product_categories = df1['Product'].unique().tolist()

st.write(product_categories)

product_categories = ['Credit card debt', 'Credit reporting', 'Conventional home mortgage', 'Checking account', 'Domestic (US) money transfer', 'FHA mortgage', 'Credit repair services',
 'Other type of mortgage', 'General-purpose credit card or charge card', 'Home equity loan or line of credit (HELOC)', 'Loan', 'Other debt', 'General-purpose prepaid card',
 'Lease', 'Medical', 'Personal line of credit', 'Other personal consumer report', 'Private student loan', 'Conventional fixed mortgage', 'Medical debt', 'Mobile or digital wallet',
 'I do not know', 'Other bank product/service', 'International money transfer', 'Vehicle loan', 'Other (i.e. phone, health club, etc.)', 'Credit card', 'VA mortgage',
 'Payday loan', 'Store credit card', 'Federal student loan servicing', 'Savings account', 'CD (Certificate of Deposit)', 'Installment loan', 'Other mortgage',
 'Other banking product or service', 'Vehicle lease', 'Auto debt', 'Conventional adjustable mortgage (ARM)', 'Payday loan debt', 'Virtual currency', 'Reverse mortgage',
 'Federal student loan debt', "Traveler's check or cashier's check", 'Cashing a check without an account', 'Payroll card', 'Non-federal student loan', 'Government benefit card',
 'Mortgage debt', 'Private student loan debt', 'Title loan', 'ID prepaid card', 'Federal student loan', 'Auto', 'Home equity loan or line of credit', 'Mortgage',
 'Debt settlement', 'General purpose card', 'Government benefit payment card', 'Mobile wallet', 'Check cashing', 'Money order', '(CD) Certificate of deposit', 'Check cashing service',
 'Pawn loan', 'Refund anticipation check', 'Telecommunications debt', 'Rental debt', 'Gift or merchant card', 'Foreign currency exchange', 'Gift card', 'Credit repair',
 'Other special purpose card', 'Transit card', 'Traveler’s/Cashier’s checks', 'Electronic Benefit Transfer / EBT card', "Money order, traveler's check or cashier's check",
 'USDA mortgage', 'Mortgage modification or foreclosure avoidance', 'Manufactured home loan', 'Student prepaid card', 'Other advances of future income', 'Student loan debt relief',
 'Earned wage access', 'Tax refund anticipation loan or check']

#st.write(f"Categories: {product_categories}")



### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run first time the app is loaded.
if "memory" not in st.session_state: ### IMPORTANT.
    model_type="gpt-4o-mini"

    # initialize the momory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True) ### IMPORTANT to use st.session_state.memory.

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # tools
    from langchain.agents import tool
    from datetime import date
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date.""" #This is the desciption the agent uses to determine whether to use the time tool.
        return "Today is " + str(date.today())

    tools = [datetoday]
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. Wait for the user to describe their problem. Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. Kindly inform the user that a ticket has been created, provide the category assigned to their complaint, and reassure them that the issue will be forwarded to the appropriate support team, who will reach out to them shortly. Maintain a professional and empathetic tone throughout."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, verbose= True)  # ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.


# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    # if (message.type in ["ai", "human"]):
    st.chat_message(message.type).write(message.content)

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("How can I help?"):
    
    # question
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input": prompt})['output']
    
    # response
    st.chat_message("assistant").write(response)
    # st.write(st.session_state.memory.buffer)


classified_data = []

# Classify by Product

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

# Append results to classified_data
classified_data.append({
    "Complaint": client_complaint,
    "Assigned Product": assigned_product,
    "Assigned Sub-product": assigned_subproduct,
    "Assigned Issue": assigned_issue
})