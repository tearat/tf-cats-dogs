import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from google.colab import files

SIZE = 224

train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)

for img, label in train[0].take(1):
    plt.figure()
    plt.imshow(img)
    print(label)

def resize_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img, label

train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False

model = tf.keras.Sequential([
    base_layers,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_batches, epochs=1)

from PIL import Image
import requests
from io import BytesIO

urls = [
    'https://cs10.pikabu.ru/post_img/big/2018/01/16/10/1516120082140277586.jpg',
    'https://www.meme-arsenal.com/memes/97f8cc0a7797857f4b5eeb44aa72f8ca.jpg',
    'https://sun9-48.userapi.com/pmy3pu0yWDilPL80sh028Vuwrt2wBUOPqtv_1A/fDhmWzz4fCk.jpg',
    'https://www.meme-arsenal.com/memes/9b0a49f5cbaa328200be55f074321b41.jpg',
    'https://cs5.pikabu.ru/images/previews_comm/2015-02_2/14232273811798.jpg',
    'https://ru.meming.world/images/ru/thumb/7/78/%D0%A8%D0%B0%D0%B1%D0%BB%D0%BE%D0%BD_%D0%BA%D0%BE%D1%82_3.jpg/300px-%D0%A8%D0%B0%D0%B1%D0%BB%D0%BE%D0%BD_%D0%BA%D0%BE%D1%82_3.jpg',
    'https://medialeaks.ru/wp-content/uploads/2019/08/glavnaya-1.jpg'
]
imgs = []
for url in urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    imgs.append(img) 

for img in imgs:
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{pred_label} {prediction}')