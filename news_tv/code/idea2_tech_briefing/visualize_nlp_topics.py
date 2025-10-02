
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows의 Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# NLP 토픽 모델링 결과 데이터
topics = {
    '토픽 #1: 정부/국제 관계': '지역, 중국, 정부, 코로나, 지원, 디지털, 영국, 미국',
    '토픽 #2: 저널리즘/취재 과정': '알다, 취재, 우리, 사람, 생각',
    '토픽 #3: 미디어와 기술': '뉴스, AI, 언론사, 데이터, 콘텐츠, 기술, 플랫폼',
    '토픽 #4: 정치/사회 뉴스 보도': '보도, 정보, 뉴스, 정치, 사실, 사회',
    '토픽 #5: 미디어 비즈니스/유통': '콘텐츠, 광고, 서비스, 플랫폼, 유튜브, 시장, 구독'
}

# 시각화
fig, ax = plt.subplots(figsize=(12, 7))

y_pos = range(len(topics))

# 각 토픽에 대한 막대 생성
ax.barh(y_pos, [1]*len(topics), align='center', color='cornflowerblue', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(topics.keys(), fontsize=12)
ax.invert_yaxis()  # labels read top-to-bottom

# 제목과 라벨 설정
ax.set_title('기사 본문 핵심 주제 (NLP 토픽 모델링)', fontsize=16, pad=20)
ax.set_xlabel('주요 키워드', fontsize=12)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # x축 눈금 및 라벨 숨기기

# 각 막대 옆에 키워드 추가
for i, (topic, keywords) in enumerate(topics.items()):
    ax.text(0.05, i, f'{keywords}', va='center', ha='left', fontsize=11, color='black')

plt.tight_layout()

# 파일로 저장
output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea2_nlp_topics.png"
plt.savefig(output_path)

print(f"시각화 자료가 {output_path} 에 저장되었습니다.")
