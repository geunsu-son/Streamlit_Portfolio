import streamlit as st
from PIL import Image
import pytesseract
import cv2
import re
import numpy as np
import pandas as pd
from io import BytesIO
import shutil

# streamlit config
st.set_page_config(
    page_title="LoL Score Board Img to Text",
    page_icon="📝",
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

def ocr_process(image):
    # 이미지를 numpy 배열로 변환
    np_image = np.array(image)

    # 이미지를 회색조로 변환
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # INTER_CUBIC을 사용하여 이미지의 크기를 변경
    resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 팽창과 침식을 위한 커널 생성
    kernel = np.ones((2, 2), np.uint8)

    # 이미지 침식
    eroded_image = cv2.erode(resized_image, kernel, iterations=1)

    # pytesseract를 사용하여 OCR 적용
    config = '--psm 4'
    text = pytesseract.image_to_string(eroded_image, lang='kor+eng', config=config)

    # 이미지와 텍스트 반환
    return eroded_image, text

# 데이터프레임을 엑셀 파일로 변환하는 함수
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    return output.getvalue()

# add streamlit title
st.title("📝 LoL Score Board Img to Text")
st.write(pytesseract.get_tesseract_version())

# 이미지 삽입 형태 예시
st.write(
    """
아래 이미지를 참고해 롤 스코어 보드를 캡처한 이미지를 넣어주세요.  
OCR의 버전이 5.0 이상이 아니라면 변환이 원활하지 않을 수 있습니다.
    """
)

uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    if st.button("텍스트 추출"):
        processed_image, text = ocr_process(image)
        text_fix = re.sub(r"[^0-9/a-zA-Z\uAC00-\uD7A3\n\s]", "", text.replace('\n\n','\n'))
        output = pd.DataFrame(columns = ['category']+['player' + str(x+1) for x in range(10)])
        text_fix_line = text_fix.split('\n')
        try:
            for text_fix_ in text_fix_line:
                if len(text_fix_) < 20:
                    continue

                text_cate_ = text_fix_[:text_fix_.find('    ')].strip()
                text_data_ = text_fix_[text_fix_.find('    '):].strip()

                text_data_ = text_data_.split(' ')
                if len(text_data_) < 10:
                    continue

                value_to_remove = ''
                while value_to_remove in text_data_:
                    text_data_.remove(value_to_remove)

                text_input_ = [text_cate_] + [x.strip() for x in text_data_]
                output.loc[len(output)] = text_input_

            output = output.transpose()
            output.columns = output.loc['category'].tolist()
            output = output.drop('category')
        
            # 스트림릿 앱에서 데이터프레임과 다운로드 버튼 표시
            st.header("추출 결과")
            st.dataframe(output)

            # 다운로드 버튼
            st.download_button(
                label="Download Excel file",
                data=to_excel(output),
                file_name='image_to_text_result_excel.xlsx',
                mime='application/vnd.ms-excel'
            )
        except:
            st.error('스코어 보드를 표로 만드는 과정에서 에러가 발생했습니다. 표로 변환하기 전 변환된 text는 아래와 같습니다.', icon="🚨")
            st.write(text_fix_line)
else:
    st.subheader('예시 이미지')

    # Load the image from file
    image_path = './source/example_lol_score_board_img.png'
    st.image(image_path, caption='상단의 메뉴 및 친구 리스트 영역이 나오지 않게 캡처해주세요. 윈도우OS 기준 캡처 단축키는 [ Win+Shift+S ] 입니다.')
