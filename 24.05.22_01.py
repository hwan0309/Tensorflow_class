import tensorflow as tf
import numpy as np
from PIL import Image
import glob

# PNG 파일 경로 패턴
file_pattern = '../Web_Macro/img_model/*.png'
files = glob.glob(file_pattern)

# glob 패턴으로 찾은 파일 리스트 출력
print("Files found:", files)


# 모든 파일을 읽어서 numpy 배열로 변환
image_list = []
for file in files:
    try:
        with Image.open(file) as img:
            # 이미지 크기를 3x3로 조정
            img = img.resize((3, 3))
            # RGB 이미지인지 확인
            if img.mode == 'RGB':
                img_array = np.array(img)
                image_list.append(img_array)
            else:
                print(f"Ignoring {file} as it is not an RGB image.")
    except Exception as e:
        print(f"Error reading {file}: {e}")


# 이미지 리스트가 비어 있지 않은지 확인
if not image_list:
    raise ValueError("No images loaded successfully.")

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(3, 3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9*3, activation='relu'),
    tf.keras.layers.Reshape((3, 3))
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 입력 이미지와 타겟 이미지를 동일하게 설정하여 학습
image_array = np.array(image_list)
model.fit(image_array, image_array, epochs=10, verbose=1)
