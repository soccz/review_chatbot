import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import atexit
import os

# 현재 폴더 경로 설정
current_folder = os.getcwd()

# TOKENIZERS_PARALLELISM 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. CSV 파일 로드
file_path = os.path.join(current_folder, 'end.csv')
data = pd.read_csv(file_path)

# 2. 텍스트 정제 함수 정의
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
    return ""

# 3. 리뷰 텍스트 결합 및 정제
data['all_reviews'] = data.apply(
    lambda row: ' '.join([
        clean_text(row['naver_review1']), clean_text(row['naver_review2']),
        clean_text(row['naver_review3']), clean_text(row['naver_review4']),
        clean_text(row['naver_review5']), clean_text(row['k_review1']),
        clean_text(row['k_review2']), clean_text(row['k_review3']),
        clean_text(row['k_review4']), clean_text(row['k_review5'])
    ]), axis=1
)

# 4. 빈 리뷰 제거
data = data[data['all_reviews'].str.strip() != '']

# 5. TF-IDF 벡터화
custom_stop_words = list(ENGLISH_STOP_WORDS.union(['맛집', '추천', '음식', '식당']))
vectorizer = TfidfVectorizer(max_features=5000, stop_words=custom_stop_words)
X = vectorizer.fit_transform(data['all_reviews'])

# 6. 질문 분석 함수 정의
def extract_keywords(question):
    question = clean_text(question)
    return question

# 7. 맛집 추천 함수 정의
def recommend_restaurants(question, data, vectorizer, X, top_n=3):
    keywords = extract_keywords(question)
    question_vec = vectorizer.transform([keywords])
    similarities = cosine_similarity(question_vec, X).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

# 8. 피드백 데이터 저장 및 로드 함수 정의
feedback_file_path = os.path.join(current_folder, 'feedback_data.csv')
feedback_data = pd.DataFrame(columns=['question', 'recommended_restaurant', 'feedback'])

def save_feedback_to_file(file_path):
    global feedback_data
    feedback_data.to_csv(file_path, index=False)

def load_feedback_from_file(file_path):
    global feedback_data
    try:
        feedback_data = pd.read_csv(file_path)
    except FileNotFoundError:
        feedback_data = pd.DataFrame(columns=['question', 'recommended_restaurant', 'feedback'])

# 프로그램 종료 시 피드백 데이터를 저장
atexit.register(lambda: save_feedback_to_file(feedback_file_path))

# 프로그램 시작 시 피드백 데이터를 로드
load_feedback_from_file(feedback_file_path)

# 9. 피드백 저장 함수 정의
def save_feedback(question, recommended_restaurant, feedback):
    global feedback_data
    new_feedback = pd.DataFrame({
        'question': [question],
        'recommended_restaurant': [recommended_restaurant],
        'feedback': [feedback]
    })
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
    save_feedback_to_file(feedback_file_path)  # 저장

# 10. 모델 재학습 함수 정의
def retrain_model(feedback_data, original_data):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    
    # 긍정적 피드백만 필터링
    positive_feedback = feedback_data[feedback_data['feedback'] == 'positive']
    
    if positive_feedback.empty:
        print("재학습할 긍정적 피드백 데이터가 충분하지 않습니다.")
        return None
    
    questions = positive_feedback['question']
    recommendations = positive_feedback['recommended_restaurant']
    
    # 피드백 데이터와 원본 데이터를 합쳐서 학습
    combined_data = pd.concat([original_data, positive_feedback])
    
    # 리뷰 텍스트와 피드백 질문을 결합
    combined_data['all_reviews'] = combined_data.apply(
        lambda row: str(row['all_reviews']) + ' ' + str(row['question']), axis=1)

    # TF-IDF 벡터화 및 모델 학습
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=custom_stop_words)
    X = vectorizer.fit_transform(combined_data['all_reviews'])
    y = combined_data['feedback'].apply(lambda x: 1 if x == 'positive' else 0)  # 타겟 레이블로 피드백 사용

    if y.nunique() < 2:
        print("재학습할 데이터의 클래스 수가 부족합니다.")
        return None

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, y)

    best_model = clf.best_estimator_
    
    return make_pipeline(vectorizer, best_model)

# 11. 미리 학습된 감정 분석 모델 로드
tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')
model = AutoModelForSequenceClassification.from_pretrained('beomi/KcELECTRA-base-v2022', num_labels=2)
sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# 12. 텍스트 요약 모델 로드
summarizer = pipeline("summarization", model="ainize/kobart-news")

# 리뷰 요약 함수 정의
def summarize_reviews(reviews):
    try:
        # 리뷰 텍스트를 합치기
        if isinstance(reviews, str):
            reviews = [reviews]
        combined_reviews = ' '.join(reviews)
        
        # 리뷰 데이터가 충분한지 확인
        if len(combined_reviews) < 50:
            return "리뷰 부족"
        
        summary = summarizer(combined_reviews, max_length=60, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"리뷰 요약 중 오류 발생: {str(e)}"

# 감정 분석 함수 정의
def analyze_sentiment(feedback_text):
    result = sentiment_analysis(feedback_text)
    if result[0]['label'] == 'positive':
        return 'positive'
    else:
        return 'negative'

# 실시간 질문-응답 시스템
def real_time_question_answering():
    while True:
        # 사용자 질문 입력
        question = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        
        # 종료 조건
        if question.lower() == 'exit':
            print("시스템을 종료합니다.")
            break
        
        # 맛집 추천
        recommended_restaurants = recommend_restaurants(question, data, vectorizer, X)
        
        for idx, restaurant in recommended_restaurants.iterrows():
            # 추천 결과 출력 및 리뷰 요약
            recommended_restaurant_details = {
                "name": restaurant["name"],
                "address": restaurant["address"],
                "reviews": restaurant["all_reviews"]
            }
            
            # 리뷰를 요약하고 한국어로 출력
            review_summary = summarize_reviews(recommended_restaurant_details["reviews"])

            print("\n추천 맛집:", recommended_restaurant_details["name"])
            print("위치:", recommended_restaurant_details["address"])
            print("추천 이유:", review_summary, "\n")
            
            # 피드백 받기
            feedback_text = input("추천이 마음에 드셨나요? (간단한 이유를 적어주세요): ")
            
            # 감정 분석
            feedback = analyze_sentiment(feedback_text)
            
            # 피드백 저장
            save_feedback(question, restaurant["name"], feedback)
            print("피드백이 저장되었습니다.")
        
        # 피드백 데이터 기반 모델 재학습
        model_pipeline = retrain_model(feedback_data, data)

# 피드백 데이터 로드
load_feedback_from_file(feedback_file_path)

# 실시간 질문-응답 시스템 실행
real_time_question_answering()
