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
    page_title="SUPER-SON Resume",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

## --- Chatbot Code --- ##
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
    with st.sidebar:
        with st.chat_message(role):
            st.markdown(message)
    if save:
        save_message(message, role)

@st.experimental_dialog("Chat History", width = "large")
def paint_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

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

retriever = load_vectorstore_and_get_retriever()

## --- sidebar --- ##
with st.sidebar:
    st.page_link("home.py", label = "이력서", icon = "📄")
    st.page_link("https://super-son.tistory.com", label = "포트폴리오", icon = "📁")
    st.page_link("https://super-son.tistory.com", label = "사이드프로젝트", icon = "💡")
    st.divider()
    s_col1, s_col2 = st.columns([0.6, 0.4])
    with s_col1:
        st.markdown("# 🤖 Chatbot")
    with s_col2:
        st.write("")
        if st.button('Chat History', use_container_width=True):
            if st.session_state["messages"] == []:
                st.sidebar.error('채팅한 이력이 없습니다. 채팅 후 다시 눌러주세요.', icon='😃')
            else:
                paint_history()

    message = st.chat_input("Ask anything about Resume...")

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
                

## --- main page --- ##
col1, col2 = st.columns([0.2, 0.8],gap="medium")

with col1:
    st.image('./source/profile_photo_comic.png', use_column_width=True)
    st.write(' ')
    with st.container(border=True):
        st.markdown("##### :hammer_and_wrench: Skill.")
        skill_dict = {'Python':0.9,
                      'SQL':0.7,
                      'AI':0.4,
                      'Excel':1.0,
                      'PowerPoint':0.9}
        
        for skill_name, skill_stat in skill_dict.items():
            st.progress(skill_stat, skill_name)

    with st.container(border=True):
        st.write(
        """
        ##### Contact.
        📞 Tel. 010-4430-2279  
        📩 E-mail. [gnsu0705@gmail.com](gnsu0705@gmail.com)  
        💻 Blog. [Super-Son](https://super-son.tistory.com/)
        """
        )
with col2:
    st.write(
        """
### 손근수 &mdash; 데이터 분석가  
방문해주셔서 감사합니다. 👋  
저는 :orange[제가 가진 능력으로 다른 사람들을 돕는 것에 기쁨을 느끼는 데이터 분석가]입니다.  
저를 더 알리고 좋은 사람들과 일하고 싶어 이 사이트를 만들게 되었습니다.  

하단의 내용을 통해 제 이력을 확인하실 수 있으며,  
👈 사이드바의 :blue[챗봇]을 활용하면 더 자세한 내용을 알아볼 수 있습니다. 😃

    """
    )
    st.write(' ')
    
    tab1, tab2 = st.tabs(["경력", "학력 및 자격증"])

    with tab1:
        st.write(
            """
    #### 데이터 분석가 | 퍼플아카데미
    **전략사업부 과장/파트장**  
    2020.04 ~ 2024.03 (48개월)  

    - 학습데이터 크롤링  
        사용중인 외부 5개 학습 사이트의 데이터를 정기적으로 크롤링할 수 있도록 구축  
        
    - 데이터 분석  
        내부 데이터를 활용한 의사결정에 필요한 데이터 분석 결과 도출  
        
    - 신규 서비스 기획 및 제작 참여  
        1) 학습 평가 모델  
            학생들의 학습 결과를 기반 각 학년별 학습 평가 모델 구축  
        
        2) 추천도서  
            학생들의 리딩 데이터 및 사내 학습 가이드를 기반으로 맞춤형 도서 추천 서비스 제작  
        
    - 프로젝트 매니징  
        1) 홈페이지 리뉴얼  
            2021년 1월 신규 홈페이지 런칭 웹기획 및 일정 관리  

        2) 신규 서비스 런칭 PM  
            2024년 2월 신규 런칭 서비스 일정 관리 및 서비스 품질 관리  
    """
            )

        with tab2:
            st.write("""
    **통계학과 | 숭실대학교**  
    2012.02. - 2018.09.

    **ADsP (데이터분석 준전문가)  | 한국데이터산업진흥원**  
    2018.12
    """
            )
