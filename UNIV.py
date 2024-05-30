import pandas as pd
import numpy as np
import tensorflow as tf

# 데이터 로드 및 전처리
data = pd.read_csv('gpascore.csv')
data = data.dropna()

ydata = data['admit'].values
xdata = []

for i, rows in data.iterrows():
    xdata.append([rows['gre'], rows['gpa'], rows['rank']])  # 'rank'와 'gpa' 순서 수정

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),  # 보통 2의 제곱으로 넣음
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),  # 마지막은 0과1로 출력되어야 하기에 1로 설정
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
model.fit(np.array(xdata), np.array(ydata), epochs=1000)

# 예측
cdata = np.array([[750, 3.7, 3], [400, 2.2, 1]])  # numpy 배열로 변환
predictions = model.predict(cdata)
print(predictions)
