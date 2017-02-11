# coding: utf-8
from __future__ import unicode_literals
import numpy as np
import librosa
import os

audio_file_name = "./piano/piano_1.mp3"
# 오디오 파일 읽기
y, sr = librosa.load(audio_file_name)
# mfcc 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, n_mfcc=20)
# 약 20*길이(초)*43 정도 (초당 약 43회 추출)
# 사이즈 확인
print(mfcc.shape)
# mfcc 변화량 추출
mfcc_delta = librosa.feature.delta(mfcc)
print(mfcc_delta.shape)
# 합치기
mfcc_and_delta = np.concatenate((mfcc, mfcc_delta), axis=0)
print(mfcc_and_delta.shape)
# 축 뒤집기
mfcc_and_delta = mfcc_and_delta.T
print(mfcc_and_delta.shape)
# 최종적으로 (43*길이, 40)의 모양이 되어야 함


overall_length = mfcc_and_delta.shape[0] #43*초
#10초씩 자르기 (10*43 = 430)
current_time = 0
window_length = 10*43
X = []
Y = []
label = 2
while current_time + window_length < overall_length:
    X_part = mfcc_and_delta[current_time:current_time+window_length,:]
    X_part = X_part.flatten()
    current_time += window_length
    X.append(X_part)
    Y.append(label) # label 추가
    
#가능하면 모든 파일을 모아서 한번에 저장하는 게 나을 것 같음   
np.save('file_path.npy', X)
np.save('file_path.npy', Y)