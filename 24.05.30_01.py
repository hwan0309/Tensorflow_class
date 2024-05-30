import tensorflow as tf

tall = [190]
shose = [280]

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = tall * a + b
    return tf.square(260 - 예측값)

opt = tf.keras.optmizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(손실함수, var_list=[a,b])
    print(a,b)
