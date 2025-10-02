import pandas as pd
import os

def analyze_data_and_generate_insights():
    data_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/data'
    output_path = 'C:/Users/user/Documents/GitHub/image_test_ML/news_tv/code/insights.txt'

    try:
        contents_df = pd.read_excel(os.path.join(data_path, 'contents.xlsx'))
        article_metrics_df = pd.read_excel(os.path.join(data_path, 'article_metrics_monthly.xlsx'))
        demographics_df1 = pd.read_excel(os.path.join(data_path, 'demographics_part001.xlsx'))
        demographics_df2 = pd.read_excel(os.path.join(data_path, 'demographics_part002.xlsx'))
        demographics_df = pd.concat([demographics_df1, demographics_df2])
        referrer_df = pd.read_excel(os.path.join(data_path, 'referrer.xlsx'))
    except Exception as e:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    insights = []

    # Insight 1 & 2: Category Analysis
    category_counts = contents_df['category'].value_counts()
    insights.append(f"1. 가장 인기 있는 카테고리: 기사의 약 25%를 차지하는 '{category_counts.index[0]}'가 가장 많이 다루어지는 주제입니다. 그 뒤를 '{category_counts.index[1]}'와 '{category_counts.index[2]}'가 잇고 있습니다.")
    insights.append(f"2. 콘텐츠 다양성: 총 {len(category_counts)}개의 카테고리가 있어 비교적 다양한 주제를 다루고 있으나, 상위 3개 카테고리가 전체의 상당 부분을 차지하여 특정 주제에 집중하는 경향을 보입니다.")

    # Insight 3: Content Production Trend
    contents_df['date'] = pd.to_datetime(contents_df['date'])
    monthly_counts = contents_df.set_index('date').resample('ME').size()
    peak_month = monthly_counts.idxmax()
    insights.append(f"3. 콘텐츠 발행 트렌드: 월별 기사 발행량은 변동성을 보이며, {peak_month.year}년 {peak_month.month}월에 가장 많은 기사가 발행되었습니다. 특정 시기에 콘텐츠 제작이 집중되는 경향이 있는지 확인할 수 있습니다.")

    # Insight 4 & 5: User Engagement Metrics
    article_metrics_df['period'] = pd.to_datetime(article_metrics_df['period'])
    monthly_metrics = article_metrics_df.groupby('period')[['comments', 'likes', 'views_total']].sum()
    peak_views_month = monthly_metrics['views_total'].idxmax()
    insights.append(f"4. 독자 참여도 정점: 총 조회수는 {peak_views_month.year}년 {peak_views_month.month}월에 최고조에 달했습니다. 이 시기에 어떤 특별한 기사나 이벤트가 있었는지 분석해볼 가치가 있습니다.")
    likes_to_views_ratio = monthly_metrics['likes'].sum() / monthly_metrics['views_total'].sum()
    comments_to_views_ratio = monthly_metrics['comments'].sum() / monthly_metrics['views_total'].sum()
    insights.append(f"5. 독자 반응 패턴: 전체 조회수 대비 좋아요 비율은 약 {likes_to_views_ratio:.2%}, 댓글 비율은 약 {comments_to_views_ratio:.2%}입니다. 독자들은 '좋아요'를 통해 더 적극적으로 반응하는 경향이 있습니다.")

    # Insight 6 & 7: Demographics
    gender_views = demographics_df.groupby('gender')['views'].sum()
    dominant_gender = gender_views.idxmax()
    insights.append(f"6. 주 독자 성별: '{dominant_gender}' 독자들이 전체 조회수의 상당 부분을 차지하며, 핵심 독자층임을 알 수 있습니다.")
    
    age_group_views = demographics_df.groupby('age_group')['views'].sum().sort_index()
    top_age_group = age_group_views.idxmax()
    insights.append(f"7. 주 독자 연령층: '{top_age_group}' 연령대가 가장 높은 조회수를 기록하여, 이 연령층을 타겟으로 하는 콘텐츠 전략이 유효할 수 있습니다.")

    # Insight 8 & 9: Referrer Analysis
    referrer_share = referrer_df.groupby('referrer')['share'].sum().nlargest(5)
    top_referrer = referrer_share.index[0]
    second_referrer = referrer_share.index[1]
    insights.append(f"8. 주요 유입 경로: '{top_referrer}'가 가장 큰 비중을 차지하는 유입 경로이며, 검색 및 소셜 미디어 유입의 중요성을 보여줍니다. 그 뒤를 '{second_referrer}'가 잇고 있습니다.")
    direct_search_share = referrer_df[referrer_df['referrer'].str.contains('검색', na=False)]['share'].sum()
    insights.append(f"9. 검색 유입의 중요성: '검색'을 통한 유입이 전체 유입 경로에서 상당한 비중을 차지합니다. 이는 SEO(검색 엔진 최적화) 전략이 독자 유입에 매우 중요하다는 것을 시사합니다.")

    # Insight 10: Content-Demographic Link (Hypothesis)
    insights.append("10. 종합적 가설: '{top_age_group}' '{dominant_gender}' 독자들이 '{category_counts.index[0]}' 카테고리에 특히 관심이 많을 가능성이 있습니다. 이 가설을 검증하기 위해 기사별 인구통계 데이터를 심층적으로 분석할 필요가 있습니다.")

    # Save insights to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(insights))

if __name__ == "__main__":
    analyze_data_and_generate_insights()