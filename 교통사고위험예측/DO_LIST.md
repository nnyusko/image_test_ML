### 교통사고 위험 예측 AI 경진대회 To-Do List

---

#### 1단계: 데이터 탐색 및 전처리 (EDA & Preprocessing)

- [x] **데이터 로드:**
    - `train.csv`, `test.csv` 불러오기
    - `train/A.csv`, `train/B.csv` 불러오기
    - `test/A.csv`, `test/B.csv` 불러오기
- [x] **데이터 병합:**
    - `train.csv`와 `train/A.csv`, `train/B.csv`를 `Test_id` 기준으로 병합
    - `test.csv`와 `test/A.csv`, `test/B.csv`를 `Test_id` 기준으로 병합
- [x] **데이터 탐색 (EDA):**
    - 병합된 데이터의 기본 통계량 확인
    - 결측치 확인 및 처리 전략 수립
    - Feature 분포 및 시각화
    - `Label`과의 상관관계 분석
- [x] **Feature Engineering:**
    - 인지 특성 데이터(`A.csv`, `B.csv`)를 기반으로 새로운 파생 변수 생성
    - 범주형 변수 인코딩
    - 수치형 변수 스케일링
- [x] **데이터셋 분리:**
    - 모델 검증을 위한 Train/Validation 데이터셋 분리

---

#### 2단계: 모델 개발 및 학습 (1차 평가 대비)

- [x] **베이스라인 모델 선정:**
    - LightGBM, XGBoost 등 Tabular 데이터에 강한 모델 선정
- [x] **모델 학습:**
    - 전처리된 데이터로 모델 학습
- [x] **모델 예측:**
    - 학습된 모델로 Test 데이터에 대한 예측 수행 (`submission.csv` 생성)
- [x] **성능 검증 및 개선:**
    - 교차 검증(Cross-Validation)을 통한 모델 성능 안정성 확인
    - 하이퍼파라미터 튜닝 (Optuna, GridSearchCV 등)
    - 다양한 모델 시도 및 앙상블(Ensemble) 기법 적용으로 성능 극대화

---

#### 3단계: 코드 제출 준비

- [x] **추론 파이프라인 구축:**
    - 데이터 전처리부터 예측까지 한번에 실행되는 스크립트(`main.py` 또는 `inference.py`) 작성
    - `submit.zip` 파일 구조에 맞게 코드 및 필요 파일 정리
- [x] **제출 환경 테스트:**
    - 제공된 베이스라인 코드(`baseline_submit.zip`) 구조 참고
    - 오프라인, CPU 환경에서 30분 내 추론이 완료되는지 확인
    - 필요한 라이브러리는 `requirements.txt`에 명시하여 패키지 설치 시간 10분 내 완료되도록 관리

---

#### 4단계: 보고서 작성 (2차 평가 대비)

- [ ] **데이터 분석 보고서 작성:**
    - 데이터 탐색 과정에서 발견한 인사이트 정리
    - Feature Engineering의 논리적 근거 및 과정 설명
    - 인지 특성과 교통사고 위험 간의 관계 분석 결과 제시
- [ ] **모델 개발 보고서 작성:**
    - 최종 모델의 아키텍처 및 선택 이유 설명
    - 모델 학습 과정, 하이퍼파라미터, 검증 방법론 상세히 기술
    - 모델 성능 및 한계점 명시

---
---

### 진행 상황 요약 (25년 10월 13일)

**1. 초기 모델링 및 실험 (완료):**
- **데이터 전처리:** `data_cleaning.py`를 통해 초기 데이터 정제 및 결측치 처리 (`train_cleaned.csv`, `test_cleaned.csv` 생성).
- **다양한 모델 실험:**
    - LightGBM 베이스라인 모델 학습 (Validation AUC: 0.640).
    - Optuna를 사용한 LightGBM 하이퍼파라미터 튜닝 (Validation AUC: 0.642).
    - XGBoost 모델 학습 (Validation AUC: 0.638).
    - 단순 평균 및 스태킹(Stacking) 앙상블 수행.
- **결과:** 이 과정들을 통해 `baseline_submit.csv`, `optuna_submit.csv`, `xgboost_submit.csv`, `ensemble_submit.csv`, `stacking_submit.csv` 등 다양한 예측 결과물 생성.

**2. 베이스라인 코드 기반 재구축 (완료):**
- **문제점 발견:** 사용자 피드백을 통해, 제출 파일(`script.py`) 및 폴더(`model/`) 구조가 대회 규정과 다른 것을 확인. 또한, 초기 모델의 Feature Engineering이 `정리.md`에 제시된 베이스라인보다 단순함을 인지.
- **신규 학습 파이프라인 구축:** `정리.md`의 코드를 기반으로, 더 정교한 Feature Engineering 로직을 적용한 `train_new_baseline.py` 스크립트 작성.
- **A/B 테스트 모델 분리 학습:**
    - A 검사 데이터에 대한 `lgbm_A.pkl` 모델 학습 (Validation AUC: **0.6787**).
    - B 검사 데이터에 대한 `lgbm_B.pkl` 모델 학습 (Validation AUC: **0.5891**).
    - 학습된 모델들을 `model/` 디렉터리에 저장.

**3. 최종 제출 패키지 생성 (완료):**
- **추론 스크립트 작성:** 대회 규정에 명시된 `script.py` 파일명과 `./data`, `./model`, `./output` 폴더 구조를 준수하는 추론 스크립트 작성.
- **`requirements.txt` 생성:** `lightgbm` 등 필요한 라이브러리 목록 작성.
- **오류 수정 및 검증:**
    - 제출 시 발생한 `unterminated string literal` 오류의 원인이었던 코드 내 오타를 `script.py`와 `train_new_baseline.py`에서 수정.
    - 수정된 파일이 최종 `submit.zip`에 올바르게 포함되었는지, 압축 해제 후 파일을 직접 읽어보는 방식으로 **교차 검증 완료**.
- **최종 파일 생성:** 수정된 `script.py`, `requirements.txt`, `model/` 폴더를 포함하는 최종 `submit.zip` 파일을 `submit/` 폴더 내에 생성.