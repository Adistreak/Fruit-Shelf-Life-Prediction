import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# Predicting Banana Ripeness Using CNN
''')
st.write("Web App For Predicting Shelf Life Of A Fruit Using Pre-Trained Convolution Neural Network")

file = st.file_uploader("", type=['jpg', 'png'])

# Main Function
def predict_stage(image_data, model):

    size = (416, 416)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image_array = np.array(image)
    print(image_array)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 416, 416, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = "Label"
    prediction = model.predict(data)

    classes = {0: 'freshripe', 1: 'freshunripe',
               2: 'overripe', 3: 'ripe', 4: 'rotten', 5: 'unripe'}
    max_val = np.amax(prediction[0])
    preds = np.where(prediction[0] == max_val)[0][0]
    stage = classes[preds]
    st.write(stage)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('ripeness.h5')
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        prediction = predict_stage(image, model)
