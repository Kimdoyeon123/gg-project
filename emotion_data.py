import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import os
import numpy as np

# Spotify API 인증 설정
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="0f9a6c744c494cba948ec67949d9158c",       # Spotify에서 발급받은 Client ID
    client_secret="bdaffd4b3d514ee48aae9842ecffa0cf" # Spotify에서 발급받은 Client Secret
))

# 곡의 ID를 통해 감정적 특성(audio features) 가져오기
def get_audio_features(track_id):
    features = sp.audio_features(track_id)
    return features[0] if features else None

# Spotify의 audio features를 7개의 감정으로 변환
def embed_emotions_from_audio_features(audio_features):
    tempo = audio_features['tempo']
    valence = audio_features['valence']
    energy = audio_features['energy']

    # 7가지 감정 벡터 초기화 (anger, disgust, fear, joy, neutral, sadness, surprise)
    emotions = np.zeros(7)

    # 감정 기여도를 기반으로 각 감정에 맞춰 임베딩
    emotions[0] = (energy * 0.6 + tempo * 0.4) / 200  # anger
    emotions[1] = ((1 - valence) * 0.7 + energy * 0.3)  # disgust
    emotions[2] = (1 - valence) * 0.6 + (tempo / 200) * 0.4  # fear
    emotions[3] = valence * 0.8 + energy * 0.2  # joy
    emotions[4] = (0.5 - abs(valence - 0.5)) * 2  # neutral
    emotions[5] = (1 - valence) * 0.8 + (1 - energy) * 0.2  # sadness
    emotions[6] = tempo / 200  # surprise

    return emotions.tolist()

# 기존 JSON 파일을 읽어오는 함수
def load_emotion_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # 파일이 없는 경우 빈 딕셔너리 반환
        data = {}
    return data

# 업데이트된 데이터를 JSON 파일에 저장하는 함수
def save_emotion_data(emotion_data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(emotion_data, f, indent=4)  # indent=4는 보기 좋게 들여쓰기
        print(f"데이터가 {file_path}에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"파일을 저장하는 데 오류가 발생했습니다: {e}")

# 새로운 데이터를 추가하는 함수
def update_emotion_data(track_name, new_emotions, file_path):
    # 기존 데이터 로드
    emotion_data = load_emotion_data(file_path)
    
    # 새로운 트랙의 감정 벡터를 추가 또는 기존 데이터를 업데이트
    emotion_data[track_name] = new_emotions
    print(f"트랙 {track_name}의 데이터가 업데이트되었습니다.")

    # 수정된 데이터 저장
    save_emotion_data(emotion_data, file_path)

# 플레이리스트 내 모든 곡의 트랙 ID 가져오기
def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    track_ids = [track['track']['id'] for track in tracks]
    track_names = [track['track']['name'] for track in tracks]
    return track_ids, track_names

# 플레이리스트 내 모든 트랙에 대해 감정 벡터를 생성하고 업데이트
def process_playlist(playlist_id, file_path):
    # 플레이리스트에서 트랙 ID와 이름 가져오기
    track_ids, track_names = get_playlist_tracks(playlist_id)

    for track_id, track_name in zip(track_ids, track_names):
        audio_features = get_audio_features(track_id)
        if audio_features:
            new_emotions = embed_emotions_from_audio_features(audio_features)
            update_emotion_data(track_name, new_emotions, file_path)
        else:
            print(f"해당 트랙에 대한 오디오 피처를 가져올 수 없습니다: {track_name} ({track_id})")

# 파일 경로 설정 (절대 경로로 확인)
file_path = os.path.abspath('gg-project/emotion_data.json')
print(f"파일 경로 확인: {file_path}")

# 예시 플레이리스트 처리
playlist_id = '1stCq3IhxqoIu7xfaWPIAc'  # Spotify의 플레이리스트 ID

# 플레이리스트 내 모든 곡 감정 분석 및 업데이트
process_playlist(playlist_id, file_path)
