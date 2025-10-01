
import pandas as pd
import numpy as np

def generate_ts_features(data, window_sizes):
    """Generates time-series (rolling) features for the given dataframe."""
    
    # 원본 데이터의 target 컬럼을 제외한 숫자형 피처만 선택
    numeric_cols = [col for col in data.columns if col.startswith('X_')]
    
    print(f"\n{len(numeric_cols)}개의 숫자형 피처에 대해 롤링 피처를 생성합니다...")
    
    # 롤링 피처 생성
    for window in window_sizes:
        print(f"  Window size: {window}...")
        # DataFrame.rolling()을 사용하여 롤링 객체 생성
        # .agg()를 사용하여 여러 통계량을 한 번에 계산
        rolling_features = data[numeric_cols].rolling(window=window, min_periods=1).agg(['mean', 'std', 'min', 'max'])
        
        # 새로운 컬럼명 생성 (예: X_01_rolling_mean_10)
        rolling_features.columns = [f'{col[0]}_roll_{col[1]}_{window}' for col in rolling_features.columns]
        
        # 원본 데이터에 새로운 피처 병합
        data = pd.concat([data, rolling_features], axis=1)
        
    return data

def main():
    print("===== 시계열 피처 엔지니어링 시작 =====")
    try:
        # 데이터 불러오기
        print("\n데이터를 불러옵니다 (train.csv, test.csv)...")
        train_df = pd.read_csv('assets/train.csv')
        test_df = pd.read_csv('assets/test.csv')
        
        # test_df에는 target 컬럼이 없으므로, 피처 생성을 위해 임시로 추가
        test_df['target'] = -1 
        
        # train/test 데이터를 합쳐서 롤링 피처를 한 번에 계산 (데이터 경계에서의 정보 손실 방지)
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 롤링 피처 생성
        window_sizes = [10, 50] # 10개, 50개 행을 기준으로 하는 롤링 피처
        combined_df_fe = generate_ts_features(combined_df, window_sizes)
        
        # 생성된 피처의 결측치(NaN) 처리 (롤링 윈도우 초반에 발생)
        # 뒤의 값으로 채우고, 그래도 남으면 앞의 값으로 채움
        print("\n생성된 피처의 결측치(NaN)를 처리합니다...")
        combined_df_fe = combined_df_fe.fillna(method='bfill').fillna(method='ffill')

        # 다시 train/test 데이터로 분리
        train_ts_fe = combined_df_fe[combined_df_fe['target'] != -1].copy()
        test_ts_fe = combined_df_fe[combined_df_fe['target'] == -1].copy()
        
        # 임시로 추가했던 target 컬럼 제거
        test_ts_fe.drop(columns=['target'], inplace=True)
        
        # 파일로 저장
        output_train_path = 'assets/train_ts_fe.csv'
        output_test_path = 'assets/test_ts_fe.csv'
        print(f"\n피처가 추가된 파일을 저장합니다...\n  - {output_train_path}\n  - {output_test_path}")
        train_ts_fe.to_csv(output_train_path, index=False)
        test_ts_fe.to_csv(output_test_path, index=False)

        print("\n===== 시계열 피처 엔지니어링 완료 =====")
        print(f"최종 Train 데이터 형태: {train_ts_fe.shape}")
        print(f"최종 Test 데이터 형태: {test_ts_fe.shape}")

    except FileNotFoundError:
        print("오류: 'assets/train.csv' 또는 'assets/test.csv' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
