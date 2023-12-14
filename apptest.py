import streamlit as st
from PIL import Image
from inference_tag2text import *

# !python -m streamlit run apptest.py   

import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

pretrained_path = 'pretrained/tag2text_swin_14m.pth'

def main():
    st.title("Image Inference App")

    # Select Model and Task
    model = st.selectbox("Select Model", ["Tag2Text", "RAM"])

    # Upload Image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Get the file path from the uploaded_file object
        image_path = save_uploaded_file(uploaded_file)

        # Run inference
        # run_inference(model, task, image_path=image_path)

    # OR Capture from Camera
    if st.button("Capture from Camera"):
        st.text("Not implemented in this example")  # Add camera capture code here

    # Run Inference Button
    if st.button("Run Inference"):
        if uploaded_file is not None:
            # Get the file path from the uploaded_file object
            image_path = save_uploaded_file(uploaded_file)

            # Run inference
            run_inference(model, task, image_path=image_path)
        else:
            st.text("Please upload an image.")

def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary location
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return uploaded_file.name

def run_inference(model, task, image_path=None):
    st.text('Running OCR inference on one image...')
    reader = easyocr.Reader(['en','hi'])
    result_txt = reader.readtext(image_path, paragraph="True", detail=0)
    result_txt = "".join(result_txt)
    if model == "Tag2Text":
        st.text('Running Tag2Text inference on one image...')
        res =  run_tag2text_inference(image_path, pretrained_path)
        # Display results
        if res:
            st.text("Tags: " + res[0])
            st.text("Description: " + res[2])
            st.text("Captions: " + result_txt)
        else:
            st.text("No results.")
    else:
        st.text('Invalid model or task')

if __name__ == "__main__":
    main()

# Function to run inference
# def run_inference(model, task, image_path=None):
#     st.text('Running OCR inference on one image...')
#     reader = easyocr.Reader(['en','hi'])
#     result_txt = reader.readtext(image_path, paragraph="True", detail=0)
#     result_txt = "".join(result_txt)
#     if model == "Tag2Text" and task == "one image":
#         st.text('Running Tag2Text inference on one image...')
#         res =  run_tag2text_inference(image_path, pretrained_path)
#         # Display results
#         if res:
#             st.text("Tags: " + res[0])
#             st.text("Description: " + res[2])
#             st.text("Captions: " + result_txt)
#         else:
#             st.text("No results.")
#         # Add your inference code here
#     elif model == "Tag2Text" and task == "multiple images":
#         st.text('Running Tag2Text inference on multiple images...')
#         res =  run_tag2text_inference(image_path, pretrained_path)
#         # Display results
#         if res:
#             st.text("Tags: " + res[0])
#             st.text("Description: " + res[2])
#             st.text("Captions: " + result_txt)
            
#         else:
#             st.text("No results.")
#         # Add your inference code here
#     elif model == "RAM" and task == "one image":
#         st.text('Running RAM inference on one image...')
#         res =  run_tag2text_inference(image_path, pretrained_path)
#         # Display results
#         if res:
#             st.text("Tags: " + res[0])
#             st.text("Description: " + res[2])
#             st.text("Captions: " + result_txt)
            
#         else:
#             st.text("No results.")
#         # Add your inference code here
#     elif model == "RAM" and task == "multiple images":
#         st.text('Running RAM inference on multiple images...')
#         res =  run_tag2text_inference(image_path, pretrained_path)
#         if res:
#             st.text("Tags: " + res[0])
#             st.text("Description: " + res[2])
#             st.text("Captions: " + result_txt)
            
#         else:
#             st.text("No results.")
#         # Add your inference code here
#     else:
#         st.text('Invalid model or task')


