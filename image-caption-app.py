import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import time
import os

import warnings
warnings.filterwarnings("ignore")

# Register your custom optimizer
class CustomAdamOptimizer(tf.keras.optimizers.Adam):
    pass

tf.keras.utils.get_custom_objects()['CustomAdamOptimizer'] = CustomAdamOptimizer

#layout of the app
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon = ":camera:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
       
st.set_option('deprecation.showfileUploaderEncoding', False)

IMG_SIZE1 = (224, 224) #image size for vgg16 model
IMG_SIZE2 = (299, 299) #image size for inception model
MAX_SEQUENCE_LENGTH = 20

def preprocess_image(image):
    img = image.resize(IMG_SIZE1)
    img = np.array(img)
    # img = np.reshape(img,[1,224,224,3])
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img

all_captions = 'tokenized_captions.txt'
# Load captions from a text file
with open('tokenized_captions.txt', 'r') as file:
    all_captions = [line.strip() for line in file]

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

    # Function to generate a caption for an image
def generate_caption(model, image, tokenizer, max_length, input_text="startseq"):
    in_text = input_text
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load the VGG16 model
vgg_model = VGG16()
for layer in vgg_model.layers:
    layer._name = 'vgg_' + layer.name
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load your pre-trained image captioning model and tokenizer
captioning_model = load_model('image-text-vgg16/vgg16_model.h5', custom_objects={'CustomAdamOptimizer': CustomAdamOptimizer})


# Function to remove "startseq" and "endseq" tokens from the predicted caption
def clean_predicted_caption(caption):
    words = caption.split()
    cleaned_words = [word for word in words if word not in ['startseq', 'endseq']]
    cleaned_caption = ' '.join(cleaned_words)
    cleaned_caption = cleaned_caption.strip() + '.'
    cleaned_caption = cleaned_caption.capitalize()
    return cleaned_caption
    
#Streamlit UI
st.title("""
         Image Caption Prediction
         """
         )
uploaded_image = st.file_uploader("", label_visibility="hidden", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    progress_text = "Loading image in progress. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Loading VGG16 model...."):
        if tokenizer is not None:
            # Preprocess the image and generate a caption with VGG16
            processed_image1 = preprocess_image(image)
            image_features = vgg_model.predict(processed_image1, verbose=0)
            caption = generate_caption(captioning_model, image_features, tokenizer, max_length)
            # Clean the predicted caption
            cleaned_caption = clean_predicted_caption(caption)
            st.success('Caption generated successfully.')
            st.write("Generated Caption:", cleaned_caption)
        
        
