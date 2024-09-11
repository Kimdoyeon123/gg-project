import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API 인증 설정
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="0f9a6c744c494cba948ec67949d9158c",       # Spotify에서 발급받은 Client ID
    client_secret="bdaffd4b3d514ee48aae9842ecffa0cf" # Spotify에서 발급받은 Client Secret
))

# 곡의 ID를 통해 감정적 특성(audio features) 가져오기
def get_audio_features(track_id):
    features = sp.audio_features(track_id)
    return features[0] if features else None

# Spotify 특성을 7가지 감정으로 임베딩하는 함수
def embed_emotions_from_audio_features(audio_features):
    tempo = audio_features['tempo']
    valence = audio_features['valence']
    energy = audio_features['energy']

    # 7가지 감정 벡터 초기화
    emotions = np.zeros(7)

    # 감정 기여도를 기반으로 각 감정에 맞춰 임베딩
    emotions[0] = (energy * 0.7 + tempo * 0.3) / 200  # anger
    emotions[1] = ((1 - valence) * 0.8 + energy * 0.2)  # disgust
    emotions[2] = (1 - valence) * 0.7 + (tempo / 200) * 0.3  # fear
    emotions[3] = valence * 0.8 + energy * 0.2  # joy
    emotions[4] = (0.5 - abs(valence - 0.5)) * 2  # neutral
    emotions[5] = (1 - valence) * 0.8 + (1 - energy) * 0.2  # sadness
    emotions[6] = tempo / 200  # surprise

    return emotions

# 곡에 대한 감정 분석 예시
track_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # 예시 곡의 Spotify 트랙 ID
audio_features = get_audio_features(track_id)
emotion_vector = embed_emotions_from_audio_features(audio_features)

print("임베딩된 감정 벡터:", emotion_vector)
