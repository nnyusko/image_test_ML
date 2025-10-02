
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from konlpy.tag import Okt
import re

def topic_modeling_analysis():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/nlp_analysis_results.txt'

    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        # For performance, let's use a sample of the data first.
        # Using all data might be very slow.
        contents_df = contents_df.dropna(subset=['content']).sample(n=1000, random_state=1)
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # --- Text Pre-processing ---
    okt = Okt()
    korean_stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다', '것', '있다', '대한', '위해', '등', '수', '그', '및', '기자', '언론', '미디어', '방송', '신문']

    def preprocess_text(text):
        text = re.sub(r'[^가-힣a-zA-Z\s]', '', text) # Remove special characters
        tokens = okt.morphs(text, stem=True) # Tokenize and stem
        tokens = [word for word in tokens if word not in korean_stopwords and len(word) > 1] # Remove stopwords and single-character words
        return ' '.join(tokens)

    try:
        contents_df['processed_content'] = contents_df['content'].apply(preprocess_text)
    except Exception as e:
        return f"텍스트 전처리 중 오류가 발생했습니다. KoNLPy 라이브러리가 올바르게 설치되었는지 확인해주세요. 오류: {e}"

    # --- Topic Modeling (LDA) ---
    if contents_df['processed_content'].empty:
        return "전처리 후 분석할 데이터가 없습니다."

    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
    try:
        dtm = vectorizer.fit_transform(contents_df['processed_content'])
    except ValueError:
        return "Document-Term Matrix를 생성할 수 없습니다. 분석할 데이터가 너무 적거나, 단어들이 모두 min_df/max_df 조건에 의해 필터링되었을 수 있습니다."

    num_topics = 5 # Let's start with 5 topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    # --- Present Results ---
    feature_names = vectorizer.get_feature_names_out()
    result = "[텍스트 마이닝 1단계: 토픽 모델링 분석 결과]\n\n기사 본문을 분석하여 5개의 주요 주제 그룹을 추출했습니다.\n"

    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        result += f"\n--- 토픽 #{topic_idx + 1} ---\n"
        result += f"주요 키워드: {', '.join(top_keywords)}\n"

    # Save result to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

if __name__ == "__main__":
    analysis_result = topic_modeling_analysis()
    print(analysis_result.encode('utf-8', 'ignore').decode('utf-8'))
