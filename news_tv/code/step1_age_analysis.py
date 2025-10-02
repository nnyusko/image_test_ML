
import pandas as pd
import os

def analyze_age_demographics():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/analysis_results.txt'

    try:
        demographics_df1 = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
        demographics_df2 = pd.read_excel(os.path.join(data_path, 'demographics_part002.xlsx'))
        demographics_df = pd.concat([demographics_df1, demographics_df2])
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # 1. Re-analyze Age Demographics
    # Filter out '전체' age group
    filtered_demographics_df = demographics_df[demographics_df['age_group'] != '전체']

    # Group by age_group and sum views
    age_group_views = filtered_demographics_df.groupby('age_group')['views'].sum().sort_values(ascending=False)

    top_age_group = age_group_views.index[0]
    top_age_group_views = age_group_views.iloc[0]
    total_views = filtered_demographics_df['views'].sum()
    top_age_group_ratio = (top_age_group_views / total_views) * 100

    result = f"[1. 핵심 타겟 연령층 재분석 결과]\n'전체'를 제외한 연령대 중, '{top_age_group}'이(가) 약 {top_age_group_views:,} 뷰, 약 {top_age_group_ratio:.2f}%의 점유율로 가장 높은 조회수를 기록한 핵심 연령층입니다."

    # Save result to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

if __name__ == "__main__":
    analysis_result = analyze_age_demographics()
    print(analysis_result)
