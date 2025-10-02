import pandas as pd
import os

def analyze_peak_month():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/analysis_results.txt'

    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        article_metrics_df = pd.read_excel(os.path.join(data_path, 'article_metrics_monthly.xlsx'))
    except Exception as e:
        return f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}"

    # 2. Investigate the Viewership Peak
    # Convert period to datetime
    article_metrics_df['period'] = pd.to_datetime(article_metrics_df['period'], format='%Y-%m')

    # Filter for May 2024
    peak_month_metrics_df = article_metrics_df[article_metrics_df['period'].dt.strftime('%Y-%m') == '2024-05']

    # Sort by views_total and get top 5 articles
    top_articles_peak_month = peak_month_metrics_df.sort_values(by='views_total', ascending=False).head(5)

    # Assuming 'article_id' in metrics_df corresponds to 'post_id' in contents_df
    # Rename for merging
    contents_df.rename(columns={'post_id': 'article_id'}, inplace=True)
    
    # Merge to get article details
    top_articles_details = pd.merge(top_articles_peak_month, contents_df, on='article_id')

    result = """

[2. 2024년 5월 조회수 정점 분석 결과]
2024년 5월에 가장 높은 조회수를 기록한 상위 5개 기사의 특징은 다음과 같습니다.
"""
    for index, row in top_articles_details.iterrows():
        result += f"\n- 제목: {row['title']}\n"
        result += f"  - 카테고리: {row['category']}\n"
        result += f"  - 태그: {row['tag']}\n"
        result += f"  - 조회수: {row['views_total']:,}\n"
    
    # Append result to file
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(result)
    
    return result

if __name__ == "__main__":
    analysis_result = analyze_peak_month()
    print(analysis_result)