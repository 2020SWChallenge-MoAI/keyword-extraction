# keyext

통계적 방법을 이용한 키워드 추출기

Keyword extraction based on statistical methods

## Workflow

### 모델 빌드 과정

- raw 문서 집합
- 전처리 (띄어쓰기 교정 및 특수문자 제거)
- 형태소 분석기 이용, tokenization
- 단어 출현 빈도 및 PMI 이용 n-gram merging
- 불필요 단어 제거 (명사 및 어근, 복합단어만 남김)
- 단어 index 생성
- (문서별) TF-IDF, (문장별) co-occurence matrix 계산

### 주요 키워드 추출 과정

- TF-IDF weight가 높은 순으로 출력

### 연관 키워드 추출 과정

- raw 질의 키워드 집합
- 각 키워드별 index로 변환. 이때 키워드를 포함하는 다른 단어까지 함께 index 집합에 포함됨 (트럼프-> id(트럼프), id(트럼프 대통령), id(도널드 트럼프), ...)
- 각 키워드별 co-occurence vector 가져온 후 합산
- weight가 높은 순으로 출력

## Installation

```sh
python setup.py install
```

You can also install by `pip` command:

```sh
pip install .
```

### Versioned Installation

```sh
python setup-versioned.py install
```
