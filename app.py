from flask import Flask, request, render_template, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# 모델 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt", return_all_scores=True)

# JSON 파일로부터 음악 감정 벡터를 불러오기
def load_music_vectors(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # JSON에서 읽어온 데이터를 np.array로 변환
    music_vectors = {track: np.array(emotions) for track, emotions in data.items()}
    return music_vectors

# 감정 벡터를 저장한 JSON 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 경로
emotion_data_file = os.path.join(current_dir, 'emotion_data.json')
music_vectors = load_music_vectors(emotion_data_file)

# 업로드된 이미지 저장 경로 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 코사인 유사도를 계산하는 함수
def calculate_cosine_similarity(emotion_vector, music_vectors):
    similarities = []

    # 감정 벡터 크기 확인
    print(f"Emotion Vector Shape: {emotion_vector.shape}")

    for music, vector in music_vectors.items():
        print(f"Music Vector Shape for {music}: {vector.shape}")  # 음악 벡터의 크기 출력
        
        if len(emotion_vector) != len(vector):
            print(f"Vector size mismatch for {music}. Skipping this track.")
            continue
        
        try:
            # 코사인 유사도 계산
            similarity = cosine_similarity([emotion_vector], [vector])[0][0]
            similarities.append((music, similarity))  # 유사도 저장
        except Exception as e:
            print(f"Error calculating cosine similarity for {music}: {e}")

    # 유사도 내림차순으로 정렬
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # 상위 5개만 터미널에 출력
    print("상위 5개의 유사도:")
    for i, (music, similarity) in enumerate(similarities[:5]):
        print(f"음악: {music}, 유사도: {similarity}")

    # 상위 5개의 음악을 반환
    return similarities[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
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

        # 감정 결과를 dict로 변환 (리스트의 첫 번째 요소에 접근)
        emotion_scores = {emotion['label']: emotion['score'] for emotion in emotions[0]}
        print("감정 점수:", emotion_scores)  # 감정 점수를 출력

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
        top_5_similarities = calculate_cosine_similarity(emotion_vector, music_vectors)

        return jsonify({
            "top_5_music": [{"music": music, "similarity": similarity} for music, similarity in top_5_similarities],
            "caption": caption,
            "emotion_scores": emotion_scores
        })

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
