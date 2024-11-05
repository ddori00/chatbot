import json
import random
from konlpy.tag import Okt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from difflib import SequenceMatcher
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 데이터 로드
with open('intents.json', encoding='utf-8') as file:    
    data = json.load(file)

# 한국어 형태소 분석기 초기화
okt = Okt()

words = []
labels = []
docs_x = []
docs_y = []

# 데이터 전처리
for intent in data['intents']:
    for pattern in intent['patterns']:
        # 토큰화 및 형태소 분석
        tokens = okt.morphs(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# 단어 중복 제거 및 정렬
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    for w in words:
        if w in doc:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# 네트워크 입력 데이터 정의
net_input = tf.keras.Input(shape=(len(training[0]),))
# fully connected 레이어 추가
net = tf.keras.layers.Dense(8, activation='relu')(net_input)
net = tf.keras.layers.Dense(8, activation='relu')(net)
# 출력 레이어 추가
net_output = tf.keras.layers.Dense(len(output[0]), activation='softmax')(net)
# 모델 정의
model = tf.keras.Model(inputs=net_input, outputs=net_output)
# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 모델 훈련
model.fit(training, output, epochs=1500, batch_size=8, verbose=1)
# 모델 저장
model.save("model_tf2.keras")