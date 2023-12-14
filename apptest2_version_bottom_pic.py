import streamlit as st
from PIL import Image
from inference_tag2text import *
# !python -m streamlit run apptest2.py 
import os
import urllib.request
import urllib
import easyocr
import cv2

# st.set_page_config(page_title="Your App Title", page_icon="✅", layout="wide", theme="light")
# Custom CSS to set a bright theme
custom_css = """
<style>
body {
    color: #000000; /* Text color */
    background-color: #FFFFFF; /* Background color */
}
</style>
"""

# Inject custom CSS
# st.markdown(custom_css, unsafe_allow_html=True)

if not os.path.exists('pretrained'):
    os.makedirs('pretrained')

pretrained_path = 'pretrained/tag2text_swin_14m.pth'
if not os.path.exists(pretrained_path):
    url = 'https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/tag2text_swin_14m.pth'
    urllib.request.urlretrieve(url, pretrained_path)
    print("Tag2Text weights downloaded!")
else:
    print("Tag2Text weights already downloaded!")


def save_uploaded_file(uploaded_file):
    temp_name_file = 'delete_me_temp_file.jpg'
    with open(temp_name_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_name_file
    
# @st.cache
# @st.cache_data
def process_image(uploaded_file, include_captions=True):
    if include_captions:
        st.text('Running OCR inference on image...')
        reader = easyocr.Reader(['en', 'hi'])
        result_txt1 = reader.readtext(uploaded_file, paragraph="True", detail=0)
        result_txt_joined = "<br>".join(result_txt1) 
    else:
        result_txt_joined = "NaN"
    st.text('Getting image description and tags...')
    res = run_tag2text_inference(uploaded_file, pretrained_path)
    return {
        "tags": res[0],
        "description": res[2],
        "caption": result_txt_joined,
    }
global results

if 'results_temp' not in st.session_state:
    st.session_state.results_temp = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


results = {
        "tags": "",
        "description": "",
        "caption": "",
    }
# Initialize session state
if 'get_res_button_click' not in st.session_state:
    st.session_state.get_res_button_click = False

from deep_translator import GoogleTranslator
def translate_description(input, selected_languages='en'):
    translated = GoogleTranslator(source='auto', target=selected_languages).translate(input)   # 
    return translated

# Set layout to wide
st.set_page_config(layout="wide", page_icon="✅",initial_sidebar_state="expanded",menu_items={'Get Help': 'https://www.extremelycoolapp.com/help','Report a bug': "https://www.extremelycoolapp.com/bug", 'About': "# This is a header. This is an *extremely* cool app!" })

st.sidebar.title("Selection")
model = st.sidebar.selectbox("Select Model", ["Tag2Text", "CLIP"])
ocr_model = st.sidebar.selectbox("Select Model", ["EasyOCR", "Tessarect", "Keras OCR"])
# Create a sidebar with a slider

# Add language selection to the sidebar
selected_languages = st.sidebar.multiselect("Caption Languages", ["English", "Hindi"], default=["English","Hindi"])

# Display selected languages
# st.sidebar.write("Selected Languages:", selected_languages)
# Display a helpful hint
st.sidebar.info("You can select the model and upload an image to start the inference.")

# sidebar_value = st.sidebar.slider("Select a value for the sidebar", min_value=1, max_value=100, value=50)
# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)
# Define the layout
col1, col2 = st.columns(2)

# Section 1 in the upper half for image uploading
with col1:
    st.header("Upload Image")


    uploaded_image = st.file_uploader("Choose a picture", type=["jpg", "jpeg", "png"])
    # st.session_state.uploaded_image = None
    # print(uploaded_image)
    
    include_captions = st.checkbox("Include Captions")
    global get_res_button_click 
    col1a, col1b, col1c = st.columns([1,1,4])

    get_res_button_click = col1a.button("Get Results")
    clear_cache_btn  = col1b.button("Clear Cache")
    st.text('Note: Please upload image with unique name or just clear cache to upload same image again')

    if clear_cache_btn:
        st.session_state.clear()
        st.session_state.uploaded_image = None
        st.session_state.results_temp = None
        clear_cache_btn = False

    if get_res_button_click:
        uploaded_image_loc = save_uploaded_file(uploaded_image)
        if st.session_state.uploaded_image != uploaded_image:
            results_temp = process_image(uploaded_image_loc, include_captions)
            st.session_state.results_temp = results_temp
            st.session_state.uploaded_image = uploaded_image
            results = results_temp.copy()
            os.remove(uploaded_image_loc)
        else:
            results_temp = st.session_state.results_temp
            results = results_temp.copy()

# Section 3 in the lower half for displaying the uploaded image
# st.header("Image")
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)


# Section 2 in the upper half for processing image and displaying results
with col2:
    st.header("Results")
    # global get_res_button_click 

    # get_res_button_click = st.button("Get Results", key="get_results_button")
    # # global results
    # if get_res_button_click:
    #     # global results
    #     uploaded_image_loc = save_uploaded_file(uploaded_image)
    #     results = process_image(uploaded_image_loc, include_captions)
        
    #     os.remove(uploaded_image_loc)
    if uploaded_image is not None:
        # Process the image and get results (dummy function, replace with your actual processing function)
        # if get_res_button_click:
        if st.session_state.uploaded_image == uploaded_image:
            results_temp = st.session_state.results_temp
            results = results_temp.copy()

        if True:
            
            # Display results
            st.subheader("Tags:")
            st.write(results["tags"])
    

            st.subheader("Description:")        
            st.write(results["description"])

            # Create two columns in the same row
            col2a, col2b, col2c = st.columns(3)

            col2a.subheader("Caption:")
            st.markdown(results["caption"], unsafe_allow_html=True)
            
            # Place the dropdown menu in the second column
            lang_opts = ['English','Hindi','French']

            # selected_language = col2b.radio("Select Language", lang_opts)
            # selected_language = col2b.selectbox("Select Language", lang_opts)

            translate_button_click = col2b.button("Translate To English", key="translate_desc_button")

            if translate_button_click:
                # Perform translation based on selected_language
                # split_captions = results["caption"].split("<br>")
                try:
                    translated_description = translate_description(results["caption"], 'en')
                    st.write("**Translation:**")
                    st.write(translated_description,unsafe_allow_html=True)
                except Exception as e:
                    st.write("**Check Internet Connection**",unsafe_allow_html=True)



# import streamlit as st
# from PIL import Image
# from inference_tag2text import *

# import os
# import easyocr
# import cv2

# pretrained_path = 'pretrained/tag2text_swin_14m.pth'

# def save_uploaded_file(uploaded_file):
#     temp_name_file = 'delete_me_temp_file.jpg'
#     with open(temp_name_file, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return temp_name_file

# def main():
#     st.title("Image Inference App")

#     # Sidebar for Settings
#     st.sidebar.title("Settings")
#     model = st.sidebar.selectbox("Select Model", ["Tag2Text", "RAM"])

#     # Main Content
#     col1, col2 = st.columns([1,3])

#     with col1:
#         st.header("Options")
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#         if uploaded_file:
#             st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=800)
#             if st.button("Run Inference"):
#                 uploaded_file_path = save_uploaded_file(uploaded_file)
#                 run_inference(model, uploaded_file_path)
#                 os.remove(uploaded_file_path)
#         else:
#             st.warning("Please upload an image.")

#     with col2:
#         st.header("Inference Results")
#         if uploaded_file:
#             st.text('Running OCR inference on one image...')
#             reader = easyocr.Reader(['en', 'hi'])
#             result_txt1 = reader.readtext(uploaded_file, paragraph="True", detail=0)
            
#             if model == "Tag2Text":
#                 st.text('Running Tag2Text inference on one image...')
#                 res = run_tag2text_inference(uploaded_file, pretrained_path)

#                 if res:
#                     st.subheader("Tags:")
#                     st.write(f"{res[0]}")
#                     st.subheader("Description:")
#                     st.write(f"{res[2]}")
#                     st.subheader("Captions:")
#                     result_txt_joined = "<br>".join(result_txt1)
#                     st.markdown(result_txt_joined, unsafe_allow_html=True)
#                 else:
#                     st.warning("No results.")
#             else:
#                 st.warning('Invalid model selection.')

#     # Additional Options
#     st.sidebar.markdown("---")
#     if st.sidebar.button("Capture from Camera"):
#         st.text("Not implemented in this example")  # Add camera capture code here

#     # Display a helpful hint
#     st.sidebar.info("You can select the model and upload an image to start the inference.")

# if __name__ == "__main__":
#     main()

#####----------------------------------------------------------------------------------------------------------------------------------------------------------


# import streamlit as st
# from PIL import Image
# from inference_tag2text import *

# import os
# import easyocr
# import cv2

# pretrained_path = 'pretrained/tag2text_swin_14m.pth'

# def save_uploaded_file(uploaded_file):
#     # Save the uploaded file to a temporary location
#     # with open(uploaded_file.name, "wb") as f:
#     temp_name_file = 'delete_me_temp_file.jpg'
#     with open(temp_name_file, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return temp_name_file
#     # return uploaded_file.name

# def main():
#     st.title("Image Inference App")

#     # Sidebar for Settings
#     st.sidebar.title("Settings")
#     model = st.sidebar.selectbox("Select Model", ["Tag2Text", "RAM"])
    
#     # Main Content
#     st.sidebar.header("Options")
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         # Display the uploaded image
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
#         # Run Inference Button
#         if st.button("Run Inference"):
#             uploaded_file_path = save_uploaded_file(uploaded_file)
#             run_inference(model, uploaded_file_path)
#             os.remove(uploaded_file_path)
#     else:
#         st.warning("Please upload an image.")

#     # Add a separator for better organization
#     st.sidebar.markdown("---")

#     # Additional Options
#     if st.sidebar.button("Capture from Camera"):
#         st.text("Not implemented in this example")  # Add camera capture code here

#     # Display a helpful hint
#     st.sidebar.info("You can select the model and upload an image to start the inference.")

# def run_inference(model, uploaded_file):
#     st.text('Running OCR inference on one image...')
#     reader = easyocr.Reader(['en', 'hi'])
#     result_txt1 = reader.readtext(uploaded_file, paragraph="True", detail=0)
#     # result_txt = "\n".join(result_txt)
#     result_txt = "\n".join(map(str, result_txt1))
#     if model == "Tag2Text":
#         st.text('Running Tag2Text inference on one image...')
#         res = run_tag2text_inference(uploaded_file, pretrained_path)

#         # Display results
#         if res:
#             st.subheader("Tags:")
#             st.write(f"{res[0]}")
#             st.subheader("Description:")
#             st.write(f"{res[2]}")
#             st.subheader("Captions:")
#             # st.write(f"Captions: {result_txt}")
#             # st.write(f"Captions:\n{result_txt}")
#             # st.text(f"Captions:\n\n```{result_txt}```")
#             # st.markdown(f"**Captions:**\n{result_txt}",unsafe_allow_html=True)
#             result_txt_joined = "<br>".join(result_txt1)      
#             st.markdown(result_txt_joined, unsafe_allow_html=True)
#         else:
#             st.warning("No results.")
#     else:
#         st.warning('Invalid model selection.')



# if __name__ == "__main__":
#     main()
