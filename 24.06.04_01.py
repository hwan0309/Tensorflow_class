from tensorflow.keras.preprocessing.image import ImageDataGenerator

생성기 = ImageDataGenerator(
    rescale=1./255,
    rotation_rnage=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
트레이닝용 = 생성기.flow_from_directory(
    'image/data_set',
    class_mode='binary',
    shuffle=True,
    seed=123,
    color_mode='rgb',
    batch_size=64,
    target_size=(64,64),
)

생성기2 = ImageDataGenerator(rescale=1./255)

검증용 = 생성기2.flow_from_directory(
    '/image/data_set',
    class_mode='binary',
    shuffle=True,
    seed=123,
    color_mode='rgb',
    batch_size=64,
)