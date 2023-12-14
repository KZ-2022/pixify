import streamlit as st
from PIL import Image
from inference_tag2text import *
# !python -m streamlit run apptest2.py 
import os
import urllib.request
import urllib
import easyocr
import cv2
import pandas as pd 
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
    with st.spinner('Loading OCR Inferance...'):
        if include_captions:
            # st.text('Running OCR inference on image...')
            reader = easyocr.Reader(['en'])   #hi
            result_txt1 = reader.readtext(uploaded_file, paragraph="True", detail=0)
            result_txt_joined = "<br>".join(result_txt1) 
        else:
            result_txt_joined = "NaN"
    with st.spinner('Getting Tags and Description...'):
        # st.text('Getting image description and tags...')
        res = run_tag2text_inference(uploaded_file, pretrained_path)
    return {
        "tags": " | ".join(list(set(res[0].split(" | ")))),
        "description": res[2],
        "caption": result_txt_joined,
    }
global results

if 'results_temp' not in st.session_state:
    st.session_state.results_temp = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# if "translate_button_click" not in st.session_state:
#     st.session_state.translate_button_click = None 



results = {
        "tags": "",
        "description": "",
        "caption": "",
    }
# Initialize session state
if 'get_res_button_click' not in st.session_state:
    st.session_state.get_res_button_click = False

language_codes = {
    'English': 'en',
    'Japanese': 'ja',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese (Simplified)': 'zh-CN',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi'
}
lang_opts = ['English', 'Japanese', 'French', 'Spanish', 'German', 'Chinese (Simplified)', 'Russian', 'Arabic', 'Hindi']

from deep_translator import GoogleTranslator
def translate_description(input, selected_languages='English'):
    selected_languages = language_codes[selected_languages]
    translated = GoogleTranslator(source='auto', target=selected_languages).translate(input)   # 
    return translated

# Set layout to wide
st.set_page_config(layout="wide", page_icon="✅",initial_sidebar_state="expanded",menu_items={'Get Help': 'https://www.extremelycoolapp.com/help','Report a bug': "https://www.extremelycoolapp.com/bug", 'About': "# This is a header. This is an *extremely* cool app!" })


my_logo = Image.open('explainers_logo2.png')
st.sidebar.image(my_logo)
# st.sidebar.title("Selection")

selected_languages = st.sidebar.multiselect("Caption Languages", ["English", "French"], default=["English","French"])

# uploaded_image_sidebar = st.sidebar.file_uploader("Choose a picture", type=["jpg", "jpeg", "png"],key='sidebar')
uploaded_image = st.sidebar.file_uploader("Choose a picture", type=["jpg", "jpeg", "png"], key='sidebar')
# st.sidebar.text('Note: Please upload image with unique name or just clear cache to upload same image again')

include_captions = st.sidebar.checkbox("Include Captions")

colsb1, colsb2 = st.sidebar.columns([0.35,0.6])
# colsb1, colsb2, colsb3 = st.sidebar.columns(3)

# get_res_button_click_sb = colsb1.button("Get Results",key='sb1')
# clear_cache_btn_sb  = colsb2.button("Clear Cache",key='sb2')

get_res_button_click = colsb1.button("Get Results",key='sb1')
clear_cache_btn  = colsb2.button("Clear Cache",key='sb2')

# st.sidebar.info("You can select the model and upload an image to start the inference.")

# my_logo = Image.open('ravens_logo.png')
# st.sidebar.image(my_logo )

# Load the logo
my_logo = Image.open('ravens_logo.png')

# Display the logo at the bottom of the sidebar using custom CSS
st.sidebar.image(my_logo, use_column_width=True)



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    margin-top: -3rem;
                    margin-bottom: -4rem;
                    padding-bottom: 0rem;
            
                    # padding-left: 2rem;
                    # padding-right:2rem
                    # margin-left: 1rem;
                    # [theme]
                    # primaryColor = "#FF4B4B"
                    # backgroundColor = "#FFFFFF"
                    # textColor = "#31333F"
                    # secondaryBackgroundColor = "#F0F2F6"
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 20% !important; # Set the width to your desired value
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            height: 100%;
            padding-top: 0rem;
            margin-top: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
        section[data-testid="column"] {
    box-shadow: rgb(0 0 0 / 20%) 0px 2px 1px -1px, rgb(0 0 0 / 14%) 0px 1px 1px 0px, rgb(0 0 0 / 12%) 0px 1px 3px 0px;
    border-radius: 15px;
    padding: 5% 5% 5% 10%;
    background-color: #f0f2f6;

} 
    </style>
    """,
    unsafe_allow_html=True,
)

import plotly.express as px

df = px.data.iris()
import base64
# @st.experimental_memo
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("mounjaro1.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-color: #f0f2f6;
padding-top: 0rem;
margin-top: 0rem;
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
border-left: 20px solid #fff;

}}

[data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
background-color: #f0f2f6;
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


artistic_title_with_box = """
    <div style="text-align: center; margin-bottom:20px; padding: 0px; background-color: #ffffff; border-radius: 15px; ">
        <h1 style="color: #4A90E2 ; text-shadow: 2px 2px 4px #000000; padding: 5px; margin:0px; font-size: 45px;">Picture Interpretation and Xploration for Imagery Features</h1>
    </div>
"""
st.markdown(artistic_title_with_box, unsafe_allow_html=True)

# Section 3 in the lower half for displaying the uploaded image

col3a, col3b, col3c = st.columns([1,8,1], gap='small')
import base64
with col3b:
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image,  use_column_width=True,) #caption="Uploaded Image",
    else:
        file = open(r"minions.gif", 'rb')
        contents = file.read()
        data_url = base64.b64encode(contents).decode('utf-8-sig')
        file.close()
        # st.markdown(f'<img src="data:image/gif;base64,{data_url}>',unsafe_allow_html = True)

        # image2 = Image.open("minions.gif")
        image2 = Image.open("lets_start1.jpg")

        st.image(image2,  use_column_width=True, output_format="GIF") #caption="Uploaded Image",

# Icons from FontAwesome
like_icon = "👍"
dislike_icon = "👎"
download_icon = "📥"

# HTML and icons for like, dislike, and download buttons
button_html = f"""
<div style="text-align: center;">
    <span style="margin: 10px;">
        <button style="border: none; background-color: transparent; cursor: pointer;" onclick="alert('Liked!')">
            {like_icon} Like
        </button>
    </span>
    <span style="margin: 10px;">
        <button style="border: none; background-color: transparent; cursor: pointer;" onclick="alert('Disliked!')">
            {dislike_icon} Dislike
        </button>
    </span>
    <span style="margin: 10px;">
        <button style="border: none; background-color: transparent; cursor: pointer;" onclick="alert('Downloading...')">
            {download_icon} Download
        </button>
    </span>
</div>
"""
# Display the buttons in a Streamlit app
# st.markdown(button_html, unsafe_allow_html=True)




# Function to generate and download CSV file
def download_csv():
    # Example DataFrame
    data = {'Name': ['John', 'Jane', 'Doe'],
            'Age': [25, 30, 22],
            'City': ['New York', 'San Francisco', 'Chicago']}
    df = pd.DataFrame(data)

    # Download CSV

    try:
        results_temp = st.session_state.results_temp
        results = results_temp.copy()

        df_res = pd.DataFrame([results])
        df_res['image'] = uploaded_image.name
        csv_data = df_res.to_csv(index=False)
    except:
        csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download File</a>'
    col3b4.markdown(href, unsafe_allow_html=True)

# Icons from FontAwesome
like_icon = "👍"
dislike_icon = "👎"
download_icon = "📥"
with col3b:
    col3b1,col3b2,col3b3,col3b4,col3b5 = st.columns([1,1,2,1,3],gap='small')
    # Display buttons and trigger download on button click
    if col3b1.button(f"{like_icon} Like"):
        # col3b1.write("Liked!")
        pass

    if col3b2.button(f"{dislike_icon} Dislike"):
        # col3b2.write("Disliked!")
        pass

    if col3b3.button(f"{download_icon} Download"):
        download_csv()

    selected_language_trans = col3b5.selectbox("Translation Language", lang_opts,key='trans_all')
    

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




col1, col2 = st.columns(2)

# Section 2 in the upper half for processing image and displaying results
with col1:
    # st.header("Results")
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

            df_res = pd.DataFrame([results])
            df_res['image'] = uploaded_image.name

        if True:
            # Display results
            st.subheader("Tags:")
            # st.write(results["tags"])

            tags_html = f"""
                <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                    <h5>{results["tags"]}</h5>
                </div>
            """
            st.markdown(tags_html, unsafe_allow_html=True)

            if results:
                # Perform translation based on selected_language
                # split_captions = results["caption"].split("<br>")
                try:
                    translated_description = translate_description(results["tags"], selected_language_trans)
                    st.write("**Translation:**")
                    # st.write(translated_description,unsafe_allow_html=True)
                    results[f'tags_{selected_language_trans}'] = translated_description

                    translated_description = f"""
                        <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                            <h5>{translated_description}</h5>
                        </div>
                    """
                    st.markdown(translated_description, unsafe_allow_html=True)


                except Exception as e:
                    # st.write("**Check Internet Connection**",unsafe_allow_html=True)
                    pass

            st.subheader("Caption:")
            # st.markdown(results["caption"], unsafe_allow_html=True)

            caption_html = f"""
                <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                    <h5>{results["caption"]}</h5>
                </div>
            """
            st.markdown(caption_html, unsafe_allow_html=True)
            
            # Place the dropdown menu in the second column
            col1a, col1b, col1c = st.columns(3)

            # lang_opts = ['English','Japanese','French']

            # selected_language = col2b.radio("Select Language", lang_opts)
            # selected_language_cap = col1a.selectbox("Select Language", lang_opts,key='caption')
            # translate_button_click = col1b.button("Translate", key="translate_cap_button")
            # translate_button_click = translate_button__cap
            # if translate_button__cap:
            # if translate_button_click:
            if results:
                # Perform translation based on selected_language
                # split_captions = results["caption"].split("<br>")
                try:
                    translated_description = translate_description(results["caption"], selected_language_trans)
                    st.write("**Translation:**")
                    # st.write(translated_description,unsafe_allow_html=True)
                    results[f'caption_{selected_language_trans}'] = translated_description
                    translated_description = f"""
                        <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                            <h5>{translated_description}</h5>
                        </div>
                    """
                    st.markdown(translated_description, unsafe_allow_html=True)
                except Exception as e:
                    # st.write("**Check Internet Connection**",unsafe_allow_html=True)
                    pass

    else:
        results_temp = {"tags": "text | stamp | paper | marker | crayon | red | white",
                        "description": "words written on a white paper with red marker",
                        "caption": "LET'S GET STARTED"}       
        st.subheader("Tags:")
        st.write(results_temp["tags"])

        st.subheader("Caption:")
        st.markdown(results_temp["caption"], unsafe_allow_html=True)


col2.write(
    """
    <style>
        {
            padding: 15px;
            background-color: #f0f2f6;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
    )

# Section 2 in the upper half for processing image and displaying results
with col2:
    if uploaded_image is not None:

        if st.session_state.uploaded_image == uploaded_image:
            results_temp = st.session_state.results_temp
            results = results_temp.copy()

        if True:
            st.subheader("Description:")        
            # st.write(results["description"])

            description_html = f"""
                <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                    <h5>{results["description"]}</h5>
                </div>
            """
            st.markdown(description_html, unsafe_allow_html=True)

            # Create two columns in the same row
            col2a, col2b, col2c = st.columns(3)
            
            # Place the dropdown menu in the second column
            # lang_opts = ['English','Japanese','French']

            # translate_desc_lang = col2a.selectbox("Select Language", lang_opts)
            # translate_button_click = col2b.button("Translate", key="translate_desc_button")
            # translate_button_click = st.session_state.translate_button_click
            # if "translate_button_click" not in st.session_state:
            #     st.session_state.translate_button_click = translate_button_click 

            # if translate_button_click:
            if results:
                # st.session_state.translate_button_click = translate_button_click 

                # Perform translation based on selected_language
                # split_captions = results["caption"].split("<br>")
                try:
                #     try:
                #         if st.session_state.uploaded_image == uploaded_image:
                #             results_temp = st.session_state.results_temp
                #             translated_description = results_temp['description' + translate_desc_lang]
                #             results = results_temp.copy()
                #         else:
                #             translated_description = translate_description(results["description"], translate_desc_lang)
                #             results_temp['description' + translate_desc_lang] = translated_description
                #             st.session_state.results_temp = results_temp
                #             results = results_temp.copy()

                #     except:
                #             translated_description = translate_description(results["description"], translate_desc_lang)
                #             results_temp['description' + translate_desc_lang] = translated_description
                #             st.session_state.results_temp = results_temp
                #             results = results_temp.copy()

                    translated_description = translate_description(results["description"], selected_language_trans)
                    st.write("**Translation:**")
                    # st.write(translated_description, unsafe_allow_html=True)
                    results[f'description_{selected_language_trans}'] = translated_description

                    translated_description = f"""
                        <div style="text-align: left; margin-bottom:0px; padding-left: 5px; background-color: #ffffff; border-radius: 15px; ">
                            <h5>{translated_description}</h5>
                        </div>
                    """
                    st.markdown(translated_description, unsafe_allow_html=True)
                except Exception as e:
                    # st.write(e)
                    # st.write("**Check Internet Connection**",unsafe_allow_html=True)
                    pass

    else:
        results_temp = {"tags": "text | stamp | paper | marker | crayon | red | white",
                        "description": "words written on a white paper with red marker",
                        "caption": "LET'S GET STARTED"}       

        st.subheader("Description:")        
        st.write(results_temp["description"])

# if st.session_state.uploaded_image == uploaded_image:
#     st.session_state.results_temp = results 

hide_streamlit_style = """
<style>
    .sub_div {
        position: absolute;
        bottom: 0px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:'Developed by Kapil and Tirthankar @ AADS-LCCI'; 
        visibility: visible;
        display: block;
        position: center;
        padding: 10px;
        top: 5px;
        # left: 600px;
    }
    div.abs{
        position: fixed;
        bottom: 0px;
        right: 10px;
        padding: 0px;
    }
</style>
"""

imageaads = Image.open("AADS.png")
col_bot1,col_bot2 = st.columns([0.8,0.4])
col_bot2.image(imageaads,  use_column_width=True,output_format="GIF")

st.sidebar.markdown(hide_streamlit_style, unsafe_allow_html=True,)


style_image1 = """
width: auto;
max-width: 850px;
height: auto;
max-height: 750px;
display: block;
justify-content: center;
border-radius: 20%;
"""
link = "https://people.com/thmb/TzDJt_cDuFa_EShaPF1WzqC8cy0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(216x0:218x2)/michael-jordan-435-3-4fc019926b644905a27a3fc98180cc41.jpg"
# st.markdown(f'<img src="{link}" style="{style_image1}">',unsafe_allow_html=True,)

#_----------------------------------------------------------------------------------------------------------------------------------------------------------
# from streamlit.components.v1 import html
# from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
# from htbuilder.units import percent, px
# from htbuilder.funcs import rgba, rgb


# def image(src_as_string, **style):
#     return img(src=src_as_string, style=styles(**style))


# def link(link, text, **style):
#     return a(_href=link, _target="_blank", style=styles(**style))(text)


# def layout(*args):
#     style = """
#     <style>
#       # MainMenu {visibility: hidden;}
#       footer {visibility: hidden;}
#      .stApp { bottom: 80px; }
#     </style>
#     """

#     style_div = styles(
#         position="fixed",
#         left=0,
#         bottom=0,
#         margin=px(0, 0, 0, 0),
#         width=percent(100),
#         color="black",
#         text_align="center",
#         height="auto",
#         opacity=1
#     )

#     # 分割线
#     style_hr = styles(
#         display="block",
#         margin=px(0, 0, 0, 0),
#         border_style="inset",
#         border_width=px(2)
#     )

#     # 修改p标签内文字的style
#     body = p(
#         id='myFooter',
#         style=styles(
#             margin=px(0, 0, 0, 0),
#             font_size="0.8rem",
#             color="rgb(51,51,51)"
#         )
#     )
#     foot = div(
#         style=style_div
#     )(
#         hr(
#             style=style_hr
#         ),
#         body
#     )

#     st.markdown(style, unsafe_allow_html=True)

#     for arg in args:
#         if isinstance(arg, str):
#             body(arg)

#         elif isinstance(arg, HtmlElement):
#             body(arg)

#     st.markdown(str(foot), unsafe_allow_html=True)

#     # js获取背景色 由于st.markdown的html实际上存在于iframe, 所以js检索的时候需要window.parent跳出到父页面
#     # 使用getComputedStyle获取所有stApp的所有样式，从中选择bgcolor
#     js_code = '''
#     <script>
#     function rgbReverse(rgb){
#         var r = rgb[0]*0.299;
#         var g = rgb[1]*0.587;
#         var b = rgb[2]*0.114;
        
#         if ((r + g + b)/255 > 0.5){
#             return "rgb(49, 51, 63)"
#         }else{
#             return "rgb(250, 250, 250)"
#         }
        
#     };
#     var stApp_css = window.parent.document.querySelector("#root > div:nth-child(1) > div > div > div");
#     window.onload = function () {
#         var mutationObserver = new MutationObserver(function(mutations) {
#                 mutations.forEach(function(mutation) {
#                     /************************当DOM元素发送改变时执行的函数体***********************/
#                     var bgColor = window.getComputedStyle(stApp_css).backgroundColor.replace("rgb(", "").replace(")", "").split(", ");
#                     var fontColor = rgbReverse(bgColor);
#                     var pTag = window.parent.document.getElementById("myFooter");
#                     pTag.style.color = fontColor;
#                     /*********************函数体结束*****************************/
#                 });
#             });
            
#             /**Element**/
#             mutationObserver.observe(stApp_css, {
#                 attributes: true,
#                 characterData: true,
#                 childList: true,
#                 subtree: true,
#                 attributeOldValue: true,
#                 characterDataOldValue: true
#             });
#     }
    

#     </script>
#     '''
#     html(js_code)


# def footer():
#     # use relative path to show my png instead of url
#     ######
#     with open('AADS.png', 'rb') as f:
#         img_logo = f.read()
#     logo_cache = st.image(img_logo)
#     logo_cache.empty()
#     ######
#     myargs = [
#         "Made at AADS",
#         image('media/ccb3c3657680f3d36265f4183c1cedde710d7242e401ab56e34bd1eb.png',
#               width=px(25), height=px(25)),
#         " with ❤️ by ",
#         link("https://www.zhihu.com/people/Gu.meng.old-dream", "@wangnov"),
#     ]
#     layout(*myargs)


# footer()


# if __name__ == '__main__':
#     main()