
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def analyze_csv(file_path):
    """
    CSV 파일을 읽어 데이터프레임의 기본 정보를 출력하고,
    수치형 및 범주형 데이터에 대한 시각화를 생성합니다.

    Args:
        file_path (str): 분석할 CSV 파일의 경로
    """
    # CSV 파일 읽기
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    # 시각화 결과를 저장할 디렉토리 생성
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    print(f"'{file_path}' 파일 분석 결과:")
    print("\n" + "="*50 + "\n")

    # 데이터프레임 기본 정보 출력
    print("### 데이터프레임 정보 ###")
    df.info()
    print("\n" + "="*50 + "\n")

    # 수치형 데이터 통계 출력
    print("### 수치형 데이터 기술 통계 ###")
    print(df.describe())
    print("\n" + "="*50 + "\n")

    # 범주형 데이터 통계 출력
    print("### 범주형 데이터 기술 통계 ###")
    print(df.describe(include=['object']))
    print("\n" + "="*50 + "\n")

    # 수치형 데이터 시각화 (히스토그램)
    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        print("### 수치형 데이터 분포 (히스토그램) ###")
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True)
            sanitized_col = col.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(output_dir, f'{sanitized_col}_histogram.png')
            plt.savefig(save_path)
            plt.close()
            print(f"'{save_path}'에 '{col}' 히스토그램 저장 완료")
        print("\n" + "="*50 + "\n")

    # 범주형 데이터 시각화 (막대 그래프)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print("### 범주형 데이터 분포 (막대 그래프) ###")
        for col in categorical_cols:
            plt.figure(figsize=(12, 7))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f'{col} Count')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            sanitized_col = col.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(output_dir, f'{sanitized_col}_countplot.png')
            plt.savefig(save_path)
            plt.close()
            print(f"'{save_path}'에 '{col}' 막대 그래프 저장 완료")
        print("\n" + "="*50 + "\n")

    # 수치형 데이터 상관관계 시각화 (히트맵)
    if len(numerical_cols) > 1:
        print("### 수치형 데이터 상관관계 (히트맵) ###")
        correlation_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        save_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(save_path)
        plt.close()
        print(f"'{save_path}'에 상관관계 히트맵 저장 완료")
        print("\n" + "="*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV 파일 분석 및 시각화')
    parser.add_argument('file_path', type=str, help='분석할 CSV 파일의 경로')
    args = parser.parse_args()

    analyze_csv(args.file_path)
