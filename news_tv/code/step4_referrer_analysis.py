import pandas as pd
import os

def analyze_referrer():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/analysis_results.txt'

    try:
        referrer_df = pd.read_excel(os.path.join(data_path, 'referrer.xlsx'))
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # 4. Detailed Referrer Analysis
    # Filter for Google and Naver search
    search_df = referrer_df[referrer_df['referrer'].str.contains('Google|네이버 통합검색', na=False)].copy()

    # Filter out URLs from referrer_detail
    search_df['keyword'] = search_df['referrer_detail'].apply(lambda x: x if not str(x).startswith('http') else None)
    
    # Drop rows with no keyword
    search_df.dropna(subset=['keyword'], inplace=True)

    # Count keyword occurrences
    keyword_counts = search_df['keyword'].value_counts().nlargest(10)

    top_keywords_result = "\n\n[4. 상세 유입 경로 분석 결과]\n\nGoogle 및 네이버 통합검색을 통한 상위 10개 검색어:\n"
    for keyword, count in keyword_counts.items():
        top_keywords_result += f"- {keyword}: {count}회\n"

    # Append result to file
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(top_keywords_result)
    
    return top_keywords_result

if __name__ == "__main__":
    analysis_result = analyze_referrer()
    print(analysis_result.encode('utf-8', 'ignore').decode('utf-8'))