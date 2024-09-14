import torch
from transformers import pipeline

# 다중 감정 분석을 위한 모델 로드
# 'j-hartmann/emotion-english-distilroberta-base'는 세부적인 감정 분석을 위한 모델
emotion_analysis = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

text = " a group of people standing around a fountainr"

# 감정 분석 실행
results = emotion_analysis(text)

# 결과 출력
for result in results[0]:
    print(f"Emotion: {result['label']}, Score: {result['score']:.4f}")