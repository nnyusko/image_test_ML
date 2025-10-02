import pandas as pd
import os

def validate_hypothesis():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/analysis_results.txt'

    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        demographics_df1 = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
        demographics_df2 = pd.read_excel(os.path.join(data_path, 'demographics_part002.xlsx'))
        demographics_df = pd.concat([demographics_df1, demographics_df2])
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # Rename for merging
    contents_df.rename(columns={'post_id': 'article_id'}, inplace=True)
    
    # Merge dataframes
    merged_df = pd.merge(demographics_df, contents_df, on='article_id')

    # --- Question 1: Top 5 article categories for 30s female users ---
    female_30s_df = merged_df[(merged_df['gender'] == '여') & (merged_df['age_group'].isin(['30-34', '35-39']))]
    top_5_articles_female_30s = female_30s_df.sort_values(by='views', ascending=False).head(5)
    top_categories_female_30s = top_5_articles_female_30s['category'].value_counts()

    q1_result = "\n\n[3. 가설 교차 검증 및 구체화 결과]\n\n1. 30대 여성 독자들이 가장 많이 본 상위 5개 기사의 카테고리 분포:\n"
    for category, count in top_categories_female_30s.items():
        q1_result += f"- {category}: {count}개\n"

    # --- Question 2: Top age/gender group for '커버스토리' category ---
    coverstory_df = merged_df[merged_df['category'] == '커버스토리']
    # Exclude '전체' from this analysis to find the most engaged specific group
    coverstory_df_filtered = coverstory_df[coverstory_df['age_group'] != '전체']
    top_group_coverstory = coverstory_df_filtered.groupby(['age_group', 'gender'])['views'].sum().sort_values(ascending=False)
    
    top_group = top_group_coverstory.index[0]
    top_group_views = top_group_coverstory.iloc[0]

    q2_result = f"\n2. '커버스토리' 카테고리를 가장 많이 본 그룹 (개별 그룹 기준):\n- 연령/성별 그룹: {top_group[0]} / {top_group[1]}\n- 총 조회수: {top_group_views:,}"

    # Append result to file
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(q1_result + q2_result)
    
    return q1_result + q2_result

if __name__ == "__main__":
    analysis_result = validate_hypothesis()
    print(analysis_result.encode('utf-8', 'ignore').decode('utf-8'))