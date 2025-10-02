
import pandas as pd
import os

def debug_merge():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    
    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        demographics_df = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    print("--- contents.xlsx 컬럼 ---")
    print(contents_df.columns)
    print(contents_df.head())

    print("\n--- demographics_part001.xlsx 컬럼 ---")
    print(demographics_df.columns)
    print(demographics_df.head())

    # Check for common values in post_id and article_id
    common_ids = set(contents_df['post_id']).intersection(set(demographics_df['article_id']))
    print(f"\n공통 ID 개수: {len(common_ids)}")

if __name__ == "__main__":
    debug_merge()

