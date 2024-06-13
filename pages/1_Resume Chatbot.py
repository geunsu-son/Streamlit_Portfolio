from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

st.set_page_config(
    page_title="Resume_Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.write(
    """
    ### 연락처
    📞 Tel. 010-4430-2279  
    📩 E-mail. [gnsu0705@gmail.com](gnsu0705@gmail.com)  
    💻 Blog. [Super-Son](https://super-son.tistory.com/)
    """
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

llm_for_memory = ChatOpenAI(
        temperature=0.1,
        streaming=True,
    )

@st.cache_data()
def load_vectorstore_and_get_retriever():
    # Load the embeddings
    embeddings = OpenAIEmbeddings()

    # Load the vectorstore from disk
    vectorstore = FAISS.load_local(f"./.cache/faiss", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})

def load_memory(_):
    return memory_tmp.load_memory_variables({})["history"]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            반드시 주어진 '이력서'를 사용해 답하세요. 만약 답을 모른다면, 그냥 모른다고 답하세요. 다른 답을 만들지 마세요.
            
            
            이력서: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=llm_for_memory,
        max_token_limit=500,
        return_messages=True,
    )

st.title("🤖 Resume_Chatbot")

st.markdown(
    """
SUPER-SON에 대해 궁금한 점을 물어보세요!  
챗봇은 제 이력서 및 업무 경험을 토대로 답변합니다.
"""
)

retriever = load_vectorstore_and_get_retriever()
send_message("안녕하세요. SUPER-SON (손근수)에 대해 어떤 것이 궁금하신가요?", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything about your file...")
if message:
    send_message(message, "human")
    memory_tmp = st.session_state["memory"]
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(load_memory),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        response = chain.invoke(message)
        save_memory(message, response.content)