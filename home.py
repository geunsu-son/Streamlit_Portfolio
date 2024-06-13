import streamlit as st

st.set_page_config(
    page_title="SUPER-SON Portfolio",
    page_icon="😃",
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

col1, col2 = st.columns([0.2, 0.8], gap="medium")

with col1:
    st.write(' ')
    st.image('./source/profile_photo_comic.png', use_column_width=True)
    st.write(' ')
    st.write(
        """
### 기술스택
- Python
- SQL
- Excel
- Powerpoint

"""
    )

with col2:
    st.write(
        """
## SUPER-SON &mdash; 데이터 분석가입니다.  
방문해주셔서 감사합니다. 👋  
저는 :orange[제가 가진 능력으로 다른 사람들을 돕는 것에 기쁨을 느끼는 데이터 분석가]입니다.  
저를 더 알리고 좋은 사람들과 일하고 싶어 이 사이트를 만들게 되었습니다.  

사이트에서는 제 이력서, 포트폴리오 및 사이드 프로젝트를 확인하실 수 있습니다.  
* 이력에 대해 궁금한 점은 :blue[[이력서 챗봇](/Resume_Chatbot)]을 참고해주세요. 😃

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
