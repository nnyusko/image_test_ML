
import pandas as pd

# pandas 출력 옵션 설정 (모든 컬럼을 볼 수 있도록)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def explore_data():
    """Loads train.csv and prints its structure to analyze time-series characteristics."""
    print("===== 데이터 탐색 시작 =====")
    try:
        train_df = pd.read_csv('assets/train.csv')

        print("\n===== 데이터 컬럼(피처) 목록 =====")
        print(train_df.columns.tolist())
        
        print("\n===== 데이터 정보 (Info) =====")
        train_df.info()

        print("\n===== 데이터 상위 5행 =====")
        print(train_df.head())

        print("\n===== 데이터 하위 5행 =====")
        print(train_df.tail())

        # ID 컬럼이 순차적으로 정렬되어 있는지 확인
        is_id_sorted = train_df['ID'].is_monotonic_increasing
        print(f"\nID 컬럼이 순차적으로 정렬되어 있는가? : {is_id_sorted}")

        if not is_id_sorted:
            print("경고: ID 컬럼이 정렬되어 있지 않아, 데이터가 시간 순서가 아닐 수 있습니다.")

    except FileNotFoundError:
        print("오류: 'assets/train.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        
    print("\n===== 데이터 탐색 완료 =====")

if __name__ == "__main__":
    explore_data()
