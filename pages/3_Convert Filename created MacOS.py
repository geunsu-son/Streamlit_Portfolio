import streamlit as st
import unicodedata
import base64

st.set_page_config(
    page_title="맥OS 한글 파일명 자소 분리 수정 : A free online converter",
    page_icon="🔁",
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


def normalize_filename(filename, include_extension=True):
    # 파일 이름과 확장자 분리
    if "." in filename and include_extension:
        name_part, extension_part = filename.rsplit(".", 1)
        normalized_name = unicodedata.normalize('NFC', name_part) + "." + extension_part
    else:
        normalized_name = unicodedata.normalize('NFC', filename)
        if not include_extension and "." in normalized_name:
            normalized_name = normalized_name.rsplit(".", 1)[0]
    return normalized_name

def copy_text_to_clipboard_alert(text):
    return '복사 완료'

st.title('맥OS에서 제작한 한글 파일명 자소 분리 수정')

# 맥OS에서 한국어 파일 이름 저장 시 발생하는 문제 설명
st.write("""
맥OS에서 한국어 파일 이름이 자음과 모음으로 분리되는 것은 문자 인코딩 문제 때문이에요.  
자모 분리 현상을 해결하고 파일의 이름을 올바르게 변환하고 싶다면 아래 기능을 활용해보세요!
1. **변환할 파일을 업로드한다.**  
    \* 업로드한 파일은 :orange[저장되지 않습니다.]  
    \* 여러 개의 파일을 업로드할 수 있습니다.  
    \* 업로드 용량 제한 : :red[**200MB**]  
         
2. **변환된 파일을 다운로드한다.**  
    \* 파일명만 복사해 사용할 수 있습니다.
""")

# 파일 업로드와 크기 제한 확인
uploaded_files = st.file_uploader("여러 파일을 업로드하세요. (200MB 제한, 파일 여러개 선택 가능)", accept_multiple_files=True)

if len(uploaded_files) > 0:
    if len(uploaded_files) == 1:
        uploaded_file = uploaded_files[0]
        st.title('파일명 자소분리 수정 결과')
        
        # 확장자 포함 옵션
        include_extension = st.checkbox("파일 확장자 포함", value=False)

        # 정규화된 파일 이름과 복사 버튼 제공
        col1, col2 = st.columns([0.85,0.15])

        # 파일 이름 정규화
        normalized_name = normalize_filename(uploaded_file.name, include_extension)
        col1.code(normalized_name)

        normalized_name = normalize_filename(uploaded_file.name, 1)
        bytes_data = uploaded_file.getvalue()
        col2.download_button(label="다운로드",
                            data=bytes_data,
                            file_name=normalized_name,
                            mime='application/octet-stream')
    else:
        st.title('파일명 자소분리 수정 결과')
        
        # 확장자 포함 옵션
        include_extension = st.checkbox("파일 확장자 포함", value=False)
        normalized_names = list()
        col1 = list()
        col2 = list()

        for i in range(len(uploaded_files)):
            uploaded_file = uploaded_files[i]

            # 정규화된 파일 이름과 복사 버튼 제공
            col1_tmp, col2_tmp = st.columns([0.8,0.2])
            col1.append(col1_tmp)
            col2.append(col2_tmp)

            # 파일 이름 정규화
            normalized_names.append(normalize_filename(uploaded_file.name, include_extension))
            col1[i].code(normalized_names[i])

            normalized_name = normalize_filename(uploaded_file.name, 1)
            bytes_data = uploaded_file.getvalue()
            col2[i].download_button(label="다운로드",
                                data=bytes_data,
                                file_name=normalized_name,
                                mime='application/octet-stream',
                                key = f'download_button_{i}')

else:
    # 기능 예시 이미지
    image_path = './source/example_convert_macos_filename_img.png'
    st.image(image_path, caption='변환 결과 예시 이미지')
