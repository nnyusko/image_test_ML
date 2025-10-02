
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. 기본 설정 및 데이터 로드 ---
def setup_korean_font():
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False

def load_data():
    try:
        contents = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\contents.xlsx")
        metrics = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\article_metrics_monthly.xlsx")
        referrer = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\referrer.xlsx")
        demographics1 = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\demographics_part001.xlsx")
        demographics2 = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\demographics_part002.xlsx")
        demographics = pd.concat([demographics1, demographics2], ignore_index=True)
        return contents, metrics, referrer, demographics
    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. {e}")
        return None, None, None, None

# --- 2. 아이디어별 분석 및 시각화 ---

# 아이디어 3: 검색 유입 분석
def analyze_and_visualize_search_keywords(referrer):
    search_referrers = referrer[referrer['referrer'].str.contains('검색', na=False)]
    top_keywords = search_referrers['referrer_detail'].value_counts().nlargest(10)
    
    df = top_keywords.to_frame().reset_index()
    df.columns = ['Keyword', 'Count']
    df = df.sort_values(by='Count', ascending=True)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(df['Keyword'], df['Count'], color='mediumseagreen')
    plt.title('상위 10개 검색 키워드 유입 현황', fontsize=16)
    plt.xlabel('검색 횟수', fontsize=12)
    plt.ylabel('검색어', fontsize=12)
    for bar in bars:
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center')
    plt.tight_layout()
    output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea3_top_search_keywords.png"
    plt.savefig(output_path)
    print(f"아이디어 3 시각화 저장: {output_path}")

# 아이디어 1: 연령대별 분석
def analyze_and_visualize_age_groups(demographics):
    age_group_views = demographics.groupby('age_group')['views'].sum().drop('전체', errors='ignore').sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = age_group_views.plot(kind='bar', color='skyblue')
    plt.title('연령대별 기사 조회수', fontsize=16)
    plt.xlabel('연령대', fontsize=12)
    plt.ylabel('총 조회수', fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, f'{int(bar.get_height()):,}', ha='center', va='bottom')
    plt.tight_layout()
    output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea1_age_distribution.png"
    plt.savefig(output_path)
    print(f"아이디어 1 시각화 저장: {output_path}")

# 아이디어 2: NLP 토픽 모델링
def analyze_and_visualize_nlp_topics(contents):
    # konlpy가 없어 sklearn의 LDA로 대체하여 분석
    # 실제 분석에서는 형태소 분석기(konlpy) 사용시 더 좋은 결과 기대 가능
    korean_stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다'] # 간단한 불용어 리스트
    
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=korean_stopwords)
    doc_term_matrix = vectorizer.fit_transform(contents['content'].dropna())
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f'토픽 #{idx+1}'] = ", ".join(top_words)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(topics))
    ax.barh(y_pos, [1]*len(topics), align='center', color='cornflowerblue', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics.keys(), fontsize=12)
    ax.invert_yaxis()
    ax.set_title('기사 본문 핵심 주제 (LDA 토픽 모델링)', fontsize=16, pad=20)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for i, (topic, keywords) in enumerate(topics.items()):
        ax.text(0.05, i, f'{keywords}', va='center', ha='left', fontsize=11, color='black')
    plt.tight_layout()
    output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea2_nlp_topics.png"
    plt.savefig(output_path)
    print(f"아이디어 2 시각화 저장: {output_path}")

# --- 3. 메인 실행 로직 ---
def main():
    setup_korean_font()
    contents, metrics, referrer, demographics = load_data()
    
    if referrer is not None:
        analyze_and_visualize_search_keywords(referrer)
    if demographics is not None:
        analyze_and_visualize_age_groups(demographics)
    if contents is not None:
        analyze_and_visualize_nlp_topics(contents)

if __name__ == '__main__':
    main()
