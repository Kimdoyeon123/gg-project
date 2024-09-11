import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API 인증 설정
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="your_client_id",       # Spotify에서 발급받은 Client ID
    client_secret="your_client_secret" # Spotify에서 발급받은 Client Secret
))

# 곡의 ID를 통해 감정적 특성(audio features) 가져오기
def get_audio_features(track_id):
    features = sp.audio_features(track_id)
    return features[0] if features else None

# 곡에 대한 감정 분석 예시
track_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # 곡의 Spotify 트랙 ID
audio_features = get_audio_features(track_id)
print(audio_features)  # 곡의 감정적 특성 (BPM, energy, valence 등) 출력

def analyze_emotion(audio_features):
    emotions = {}

    # 템포(BPM)가 높을수록 흥분
    tempo = audio_features['tempo']
    emotions['excitement'] = tempo / 200  # 예: 200 BPM을 기준으로

    # Valence가 높으면 더 긍정적
    valence = audio_features['valence']
    emotions['happiness'] = valence

    # 에너지가 높으면 활발한 곡
    energy = audio_features['energy']
    emotions['energy'] = energy

    # 슬픔은 valence가 낮고 템포가 느린 경우로 간주
    emotions['sadness'] = (1 - valence) * (1 - tempo / 200)

    return emotions


import json

# 음악 감정 데이터를 파일로 저장
def save_emotions_to_file(emotion_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(emotion_data, f)

# 예시 감정 데이터
emotion_data = {
    "Happy by Pharrell Williams": [0.1, 0.0, 0.1, 0.8, 0.0, 0.0, 0.0],
    "Good Time by Carly Rae Jepsen": [0.1, 0.0, 0.1, 0.7, 0.0, 0.0, 0.1]
}

# 감정 데이터를 JSON 파일로 저장
save_emotions_to_file(emotion_data, "gg-project/emotion_data.json")
