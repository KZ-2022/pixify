FROM python:3.11.6

EXPOSE 80

MAINTAINER KZ

WORKDIR /src
COPY requirements.txt ./requirements.txt

RUN apt-get update
#RUN apt-get install libgdiplus build-essential cmake pkg-config libx11-dev libatlas-base-dev \
#libgtk-3-dev libboost-python-dev -y
#RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install libpoppler-cpp-dev poppler-utils -y
RUN apt-get install gunicorn3 -y
# RUN apt-get install Flask -y
# RUN apt-get install deep-translator -y 
RUN pip install -r requirements.txt

COPY . .
CMD streamlit run apptest2.py \
    --browser.gatherUsageStats false\
    --server.port=80

# CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app", "--workers=5"]
# CMD ["python","app.py"]