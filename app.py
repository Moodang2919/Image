import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle


with open('model.pkl','rb') as f:
    model = pickle.load(f)
    
st.title("Image Classification with MobileNetV2 by Boonnisa Tangjitpermkwamdee")

upload_file = st.fire_uploader("Upload image",type=["jpg","png","jpeg"])

if upload_file is not None:
    #display a image on screen
    img = Image.open(upload_file)
    st.image(img, caption="upload Image")
    
    #preprocessing
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Prediction
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]
   
    st.subheader("Prediction:")
    for i, pred in enumerate(top_preds):
       st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")
