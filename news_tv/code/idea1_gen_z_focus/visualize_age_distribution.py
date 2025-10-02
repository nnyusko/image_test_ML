
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows의 Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 데이터 불러오기
try:
    df1 = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\demographics_part001.xlsx")
    df2 = pd.read_excel("C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\data\\demographics_part002.xlsx")
    df = pd.concat([df1, df2], ignore_index=True)
except FileNotFoundError:
    print("오류: demographics 엑셀 파일을 찾을 수 없습니다.")
    exit()


# 연령대별 조회수 집계
age_group_views = df.groupby('age_group')['views'].sum().sort_values(ascending=False)

# '전체' 연령대 제외
age_group_views = age_group_views.drop('전체', errors='ignore')

# 시각화
plt.figure(figsize=(10, 6))
bars = age_group_views.plot(kind='bar', color='skyblue')
plt.title('연령대별 기사 조회수', fontsize=16)
plt.xlabel('연령대', fontsize=12)
plt.ylabel('총 조회수', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# 막대 위에 값 표시
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, f'{int(bar.get_height()):,}', ha='center', va='bottom')

# 파일로 저장
output_path = "C:\\Users\\user\\Documents\\GitHub\\image_test_ML\\news_tv\\code\\visualizations\\idea1_age_distribution.png"
plt.savefig(output_path)

print(f"시각화 자료가 {output_path} 에 저장되었습니다.")
