import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# 한글 폰트 설정
# 1. 나눔고딕 폰트 설치
# sudo apt-get install -y fonts-nanum*
# 2. matplotlib의 폰트 캐시 삭제
# rm -rf ~/.cache/matplotlib/*

# 설치된 나눔글꼴중 하나를 선택
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows의 경우
# font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # Linux의 경우
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 데이터 파일 경로
data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/visualizations'

# 시각화 결과 저장 폴더 생성
os.makedirs(output_path, exist_ok=True)

# 데이터 불러오기
contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
article_metrics_df = pd.read_excel(os.path.join(data_path, 'article_metrics_monthly.xlsx'))
demographics_df1 = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
demographics_df2 = pd.read_excel(os.path.join(data_path, 'demographics_part002.xlsx'))
demographics_df = pd.concat([demographics_df1, demographics_df2])
referrer_df = pd.read_excel(os.path.join(data_path, 'referrer.xlsx'))

# 1. contents.xlsx 시각화
# 카테고리별 게시물 수
plt.figure(figsize=(12, 6))
sns.countplot(y='category', data=contents_df, order = contents_df['category'].value_counts().index)
plt.title('카테고리별 게시물 수')
plt.xlabel('게시물 수')
plt.ylabel('카테고리')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'contents_category_distribution.png'))
plt.close()

# 날짜별 게시물 수
contents_df['date'] = pd.to_datetime(contents_df['date'])
contents_df.set_index('date', inplace=True)
monthly_counts = contents_df.resample('M').size()
plt.figure(figsize=(12, 6))
monthly_counts.plot()
plt.title('월별 게시물 수')
plt.xlabel('날짜')
plt.ylabel('게시물 수')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'contents_monthly_counts.png'))
plt.close()

# 2. article_metrics_monthly.xlsx 시각화
article_metrics_df['period'] = pd.to_datetime(article_metrics_df['period'], format='%Y-%m')
monthly_metrics = article_metrics_df.groupby('period')[['comments', 'likes', 'views_total']].sum()

plt.figure(figsize=(12, 6))
monthly_metrics.plot(subplots=True, figsize=(12, 10))
plt.suptitle('월별 댓글, 좋아요, 조회수')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_path, 'article_metrics_monthly.png'))
plt.close()


# 3. demographics_df 시각화
# 성별 조회수
gender_views = demographics_df.groupby('gender')['views'].sum()
plt.figure(figsize=(8, 5))
gender_views.plot(kind='bar')
plt.title('성별에 따른 조회수')
plt.xlabel('성별')
plt.ylabel('조회수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'demographics_gender_views.png'))
plt.close()

# 연령대별 조회수
age_group_views = demographics_df.groupby('age_group')['views'].sum().sort_index()
plt.figure(figsize=(10, 6))
age_group_views.plot(kind='bar')
plt.title('연령대별 조회수')
plt.xlabel('연령대')
plt.ylabel('조회수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'demographics_age_group_views.png'))
plt.close()

# 4. referrer.xlsx 시각화
# 유입 경로별 조회수 비율
referrer_top10 = referrer_df.groupby('referrer')['share'].sum().nlargest(10)
plt.figure(figsize=(12, 8))
referrer_top10.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('상위 10개 유입 경로 (Share %)')
plt.ylabel('')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'referrer_top10_pie.png'))
plt.close()

print(f"시각화 파일이 {output_path} 에 저장되었습니다.")