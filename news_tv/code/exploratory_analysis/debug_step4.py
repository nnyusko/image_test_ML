
import pandas as pd
import os

def debug_referrer():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'

    try:
        referrer_df = pd.read_excel(os.path.join(data_path, 'referrer.xlsx'))
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # Filter for Google and Naver search
    search_df = referrer_df[referrer_df['referrer'].str.contains('Google|네이버 통합검색', na=False)]

    print("--- 검색 유입 상세 정보 샘플 ---")
    print(search_df['referrer_detail'].head(20))

if __name__ == "__main__":
    debug_referrer()
