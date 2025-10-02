
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows의 Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 주요 검색 키워드 데이터
keywords_data = {
    '신문과 방송': 664,
    '신문과방송': 554,
    '한국언론진흥재단': 547,
    '기타': 320,
    '신문과방송 블로그': 248,
    '언론진흥재단': 203,
    '저널리즘의 기본 원칙': 74,
    '미디어 용어': 71,
    '신문과 방송 블로그': 56,
    '언론사 채용': 53
}

# 데이터프레임 생성 및 정렬
df = pd.DataFrame(list(keywords_data.items()), columns=['Keyword', 'Count'])
df = df.sort_values(by='Count', ascending=True)

# 시각화
plt.figure(figsize=(12, 8))
bars = plt.barh(df['Keyword'], df['Count'], color='mediumseagreen')

plt.title('상위 10개 검색 키워드 유입 현황', fontsize=16)
plt.xlabel('검색 횟수', fontsize=12)
plt.ylabel('검색어', fontsize=12)
plt.tight_layout()

# 막대 옆에 값 표시
for bar in bars:
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center')

# 파일로 저장
output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea3_top_search_keywords.png"
plt.savefig(output_path)

print(f"시각화 자료가 {output_path} 에 저장되었습니다.")
