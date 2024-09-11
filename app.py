from flask import Flask, request, render_template, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 모델 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
emotion_classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base", framework="pt")

# 미리 저장된 음악 감정 벡터 (7차원: anger, disgust, fear, joy, neutral, sadness, surprise)
music_vectors = {
    "Happy by Pharrell Williams": np.array([0.1, 0.0, 0.1, 0.8, 0.0, 0.0, 0.0]),
    "Good Time by Carly Rae Jepsen": np.array([0.1, 0.0, 0.1, 0.7, 0.0, 0.0, 0.1]),
    "Someone Like You by Adele": np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.8, 0.0]),
    "The Night We Met by Lord Huron": np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.9, 0.0]),
    "Break Stuff by Limp Bizkit": np.array([0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "Numb by Linkin Park": np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

# 업로드된 이미지 저장 경로 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_top_two_emotions(emotion_scores):
    # 감정 결과에서 상위 두 감정 추출
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    top_two_emotions = sorted_emotions[:2]
    return top_two_emotions

def calculate_cosine_similarity(emotion_vector, music_vectors):
    # 코사인 유사도를 계산하여 가장 유사한 음악을 선택
    best_match = None
    highest_similarity = -1
    for music, vector in music_vectors.items():
        similarity = cosine_similarity([emotion_vector], [vector])[0][0]
        print(f"음악: {music}, 유사도: {similarity}")  # 각 음악에 대한 유사도 출력
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = music
    return best_match if best_match else "No music available for this emotion"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # 파일 업로드 처리
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 파일 저장
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 이미지 열기 및 처리
        image = Image.open(filepath)
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # 감정 분석
        emotions = emotion_classifier(caption)
        print("감정 분석 결과:", emotions)  # 감정 분석 결과를 터미널에 출력

        # 감정 결과를 dict로 변환
        emotion_scores = {emotion['label']: emotion['score'] for emotion in emotions}
        print("감정 점수:", emotion_scores)  # 감정 점수를 출력

        # 상위 두 감정 추출
        top_two_emotions = get_top_two_emotions(emotion_scores)

        # 감정 벡터 생성 (7차원: anger, disgust, fear, joy, neutral, sadness, surprise)
        emotion_vector = np.array([
            emotion_scores.get('anger', 0),
            emotion_scores.get('disgust', 0),
            emotion_scores.get('fear', 0),
            emotion_scores.get('joy', 0),
            emotion_scores.get('neutral', 0),
            emotion_scores.get('sadness', 0),
            emotion_scores.get('surprise', 0)
        ])
        print("감정 벡터:", emotion_vector)  # 감정 벡터를 출력

        # 음악 추천: 코사인 유사도 계산
        recommended_music = calculate_cosine_similarity(emotion_vector, music_vectors)

        return jsonify({"music": recommended_music, "caption": caption, "top_emotions": top_two_emotions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 외부 접근 가능하도록 호스트와 포트 설정
