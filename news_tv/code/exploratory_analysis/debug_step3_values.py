import pandas as pd
import os

def debug_step3_values():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    
    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        demographics_df1 = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
        demographics_df2 = pd.read_excel(os.path.join(data_path, 'demographics_part002.xlsx'))
        demographics_df = pd.concat([demographics_df1, demographics_df2])
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # Merge dataframes
    merged_df = pd.merge(demographics_df, contents_df, on='article_id')

    print("--- 성별 필터링 카운트 ---")
    print(merged_df[merged_df['gender'] == '여'].shape[0])

    print("\n--- 연령대 필터링 카운트 ---")
    print(merged_df[merged_df['age_group'] == '30-39'].shape[0])

    print("\n--- 동시 필터링 카운트 ---")
    print(merged_df[(merged_df['gender'] == '여') & (merged_df['age_group'] == '30-39')].shape[0])

if __name__ == "__main__":
    debug_step3_values()