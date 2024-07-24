import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from konlpy.tag import Okt
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.utils import resample

# 데이터 로딩 및 전처리
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path, low_memory=False)
        data = data.drop(['소재지전화', '소재지면적', '소재지우편번호', '소재지전체주소', '도로명우편번호'], axis=1)
        data['좌표정보(x)'] = data['좌표정보(x)'].astype('float32')
        data['좌표정보(y)'] = data['좌표정보(y)'].astype('float32')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    return data

# 텍스트 전처리
okt = Okt()
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = okt.morphs(text.lower())
    return ' '.join([word for word in tokens if len(word) > 1])

# 리뷰 통합
def combine_reviews(row):
    reviews = []
    for i in range(1, 6):
        naver_review = row.get(f'naver_review{i}', '')
        k_review = row.get(f'k_review{i}', '')
        reviews.extend([str(naver_review), str(k_review)])  # 문자열로 변환
    return ' '.join(filter(None, reviews))

# 데이터 균형 맞추기
def balance_classes(data, target_column):
    majority_class = data[target_column].value_counts().idxmax()
    balanced_data = data[data[target_column] == majority_class]

    for class_value in data[target_column].unique():
        if class_value != majority_class:
            class_data = data[data[target_column] == class_value]
            class_data_resampled = resample(class_data,
                                            replace=True,    # 샘플을 복원하여 샘플링
                                            n_samples=len(balanced_data),  # 다수 클래스의 크기로 맞춤
                                            random_state=42)
            balanced_data = pd.concat([balanced_data, class_data_resampled])

    return balanced_data

# Word2Vec 모델 학습
def train_word2vec(reviews):
    tokenized_reviews = [okt.morphs(review) for review in reviews if isinstance(review, str)]
    model = Word2Vec(sentences=tokenized_reviews, vector_size=150, window=5, min_count=2, workers=4)
    return model

# 문서 벡터화
def get_document_vector(doc, model):
    words = okt.morphs(doc)
    valid_words = [model.wv[word] for word in words if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean(valid_words, axis=0)

# 메인 함수
def main():
    # 데이터 로딩 및 전처리
    data = load_and_preprocess_data('end.csv')
    if data is None:
        return

    # 데이터 샘플링 (선택적으로 전체 데이터 중 일부만 사용)
    data = data.sample(frac=0.1, random_state=42)  # 데이터의 10%만 사용

    data['combined_reviews'] = data.apply(combine_reviews, axis=1)
    data['preprocessed_reviews'] = data['combined_reviews'].apply(preprocess_text)

    # 데이터 균형 맞추기
    data = balance_classes(data, '업태구분명')

    # Word2Vec 모델 학습
    w2v_model = train_word2vec(data['preprocessed_reviews'])

    # 문서 벡터화
    data['review_vector'] = data['preprocessed_reviews'].apply(lambda x: get_document_vector(x, w2v_model))

    # 특성 및 레이블 준비
    X = np.array(data['review_vector'].tolist())
    le = LabelEncoder()
    y = le.fit_transform(data['업태구분명'])

    # 학습 및 테스트 세트 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 하이퍼파라미터 튜닝
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_

    # LightGBM 하이퍼파라미터 튜닝
    param_grid_lgbm = {
        'num_leaves': [31, 50],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200]
    }

    lgbm = LGBMClassifier(random_state=42, n_jobs=1)
    grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid_lgbm, cv=3, n_jobs=1, verbose=2)
    grid_search_lgbm.fit(X_train, y_train)
    best_lgbm_model = grid_search_lgbm.best_estimator_

    # 교차 검증 및 성능 비교
    scores_rf = cross_val_score(best_rf_model, X_train, y_train, cv=3, n_jobs=1)
    scores_lgbm = cross_val_score(best_lgbm_model, X_train, y_train, cv=3, n_jobs=1)

    print(f"랜덤 포레스트 교차 검증 정확도: {np.mean(scores_rf):.2f}")
    print(f"LightGBM 교차 검증 정확도: {np.mean(scores_lgbm):.2f}")

    # 최종 모델 평가
    accuracy_rf = best_rf_model.score(X_test, y_test)
    accuracy_lgbm = best_lgbm_model.score(X_test, y_test)

    print(f"랜덤 포레스트 모델 정확도: {accuracy_rf:.2f}")
    print(f"LightGBM 모델 정확도: {accuracy_lgbm:.2f}")

    # 성능이 더 좋은 모델을 선택하여 저장
    best_model = best_rf_model if accuracy_rf > accuracy_lgbm else best_lgbm_model

    # 모델 및 관련 객체 저장
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    w2v_model.save('word2vec_model')
    data[['name', 'address', 'review_vector']].to_pickle('restaurant_data.pkl')

# 추천 시스템
def recommend_restaurant(query, top_n=5):
    try:
        # 저장된 모델 및 데이터 로드
        best_rf_model = joblib.load('best_rf_model.joblib')
        le = joblib.load('label_encoder.joblib')
        w2v_model = Word2Vec.load('word2vec_model')
        restaurant_data = pd.read_pickle('restaurant_data.pkl')

        # 쿼리 벡터화
        query_vector = get_document_vector(query, w2v_model)
        if np.all(query_vector == 0):
            print("Query is too unique, no similar words found in model.")
            return

        # 유사도 계산 및 추천
        similarities = cosine_similarity([query_vector], list(restaurant_data['review_vector']))
        top_indices = similarities[0].argsort()[-top_n:][::-1]

        recommendations = restaurant_data.iloc[top_indices]
        for _, row in recommendations.iterrows():
            print(f"식당 이름: {row['name']}")
            print(f"주소: {row['address']}")
            print("---")
    except Exception as e:
        print(f"Error in recommendation system: {e}")

if __name__ == "__main__":
    main()
    

    # 추천 시스템 사용 예시
    query = "맛있는 한식 식당 추천해주세요"
    print("추천 식당:")
    recommend_restaurant(query)
