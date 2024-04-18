import streamlit as st
from langchain.prompts import MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import get_openai_callback, StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import os

st.set_page_config(
    page_title="Lallan Lucknow AI",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
)



llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key='AIzaSyCMAvB0-ehycivbI10OaaqY9WNXUe20U7U')
# google_api_key = st.secrets['']

# json upload
def write_to_json(data, filename):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)
        # Check if the email already exists in the file
        if "email" in data:
            existing_emails = [
                entry["email"] for entry in existing_data if "email" in entry
            ]
            if data["email"] in existing_emails:
                return
        existing_data.append(data)
        with open(filename, "w") as file:
            json.dump(existing_data, file, indent=4)
    else:
        with open(filename, "w") as file:
            json.dump([data], file, indent=4)


# session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
if "doc" not in st.session_state:
    st.session_state.doc = PineconeVectorStore(
        index_name="lai-rag", embedding=st.session_state.embeddings, pinecone_api_key='71038238-854c-40ab-b350-d4d2ca3fbb13'
    )
if "inpu" not in st.session_state:
    st.session_state.inpu = False

if "email" not in st.session_state:
    st.session_state.email = ""


# form
with st.form("my_form"):
    st.header("Enter you email")
    email = st.text_input("Email")
    submitted = st.form_submit_button("Submit")
    if submitted and email != "":
        write_to_json({"email": email}, "emails.json")
        st.session_state.email = email
        st.session_state.inpu = True


#####
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
prompt_template = PromptTemplate.from_template(
    "You are an expert informator system about Lucknow, I'll give you question and context and you'll return the answer in a sweet and sarcastic tone. You will use Hum instead of main. Your name is Lallan. The full form of Lallan is 'Lucknow Artificial Language and Learning Assistance Network'. Call only Janab-e-Alaa instead of phrase My dear Friend. Say Salaam Miya! instead of Greetings. Here is the prompt\n{question}\nanswer it using the following context\n{context}."
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": st.session_state.doc.as_retriever(search_kwargs={"k": 6})
        | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)


msgs = StreamlitChatMessageHistory()
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

if prompt := st.chat_input(
    "Farmaiye Janaab", disabled=False if st.session_state.inpu == True else True
):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        a = rag_chain.invoke(prompt)
        st.spinner(text="In progress...")
        st.markdown(a)
    email_filename = os.path.join(
        "queries", f"{st.session_state.email.split('@')[0]}.json"
    )
    queries_folder = "queries"
    if not os.path.exists(queries_folder):
        os.makedirs(queries_folder)
    write_to_json(
        {"prompt": prompt, "answer": a},
        email_filename,
    )
    st.session_state.messages.append({"role": "assistant", "content": a})
