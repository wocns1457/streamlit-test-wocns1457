import streamlit as st
import requests
import io
import os
import base64
import time
import pandas as pd
from itertools import tee
from PIL import Image


server_ip = st.secrets["SERVER_IP"]
gpu_server_ip = st.secrets["GPU_SERVER_IP"]

# download video
def download_files(server_ip, file_name, save_dir):
    url = server_ip + file_name 
    local_path = os.path.join(save_dir, file_name) 

    with requests.get(url, stream=True) as r:
        r.raise_for_status() 
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # 8KB씩 다운로드
                if chunk:
                    f.write(chunk)

sample_dir = './sample'
video_files = ["sample1.mp4", "sample2.mp4", "sample3.mp4"]

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    
# for file_name in video_files:
#     local_path = os.path.join(sample_dir, file_name)
#     if not os.path.exists(local_path):
#         download_files(server_ip, file_name, sample_dir)
      
input_data = None

# Streamlit 앱 인터페이스
st.set_page_config(layout="wide")
st.title("Image Classification with Flask and Streamlit")

#sidebar
with st.sidebar:
    st.title('***Configuration***')
    st.subheader('')

    use_sample = st.radio("**Use sample data?**", 
                     ["**No**", "**Yes**"],
                     captions=["", "비디오만 적용."]
                     ).replace('*', '')

    if use_sample=='Yes':
        file_name = st.selectbox("Choose a sample video", 
                        (['sample1.mp4', 'sample2.mp4' ,'sample3.mp4']))
    st.subheader('')
    sampling_rate = st.selectbox("Set the video sampling rate", 
                        ([i for i in range(30, 0, -1)]))
    st.subheader('')
  
    num_top_k = st.selectbox("Set the num of top_k", 
                        ([i for i in range(1, 11)]))
    st.subheader('')
    conf_thres = st.slider("Set the object confidence threshold", 0.0, 1.0, 0.25)
    
    st.subheader('')
    sim_thres = st.slider("Set the text-image similarity threshold", 0.0, 1.0, 0.25)

col1, col2 = st.columns([1, 0.7], gap='large')

if use_sample == 'Yes':
    col1.subheader('')
    col1.subheader(file_name)
    video_path = os.path.join(sample_dir, file_name)
    input_data = col1.video(video_path)
    file_type = 'video'
    col1.warning("영상이 재생되지 않는 경우, 플레이어가 지원하지 않는 코덱을 사용하고 있을 것입니다.")

elif use_sample == 'No':
    uploaded_file = col1.file_uploader("Choose an image...", 
                                       type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'wmv'])
    file_name = uploaded_file.name if uploaded_file is not None else None
    file_type = uploaded_file.type.split('/')[0] if uploaded_file is not None else None

    with col1:
        if file_type == 'image':
            num_top_k = 1
            sampling_rate = 1
            input_data = st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        elif file_type == 'video':
            input_data = st.video(uploaded_file)
            st.warning("영상이 재생되지 않는 경우, 플레이어가 지원하지 않는 코덱을 사용하고 있을 것입니다.")

if input_data:
    with col2:
        st.subheader('')
        st.subheader('')
        prompt = st.text_area('Write your prompt', 
                              'example..\n'
                              'he wearing black shirt...',
                               height=68)
        button = st.button('Run Model', use_container_width=True)

    # Flask API에 이미지 전송
    if button and prompt:
        col2.subheader('Configuration Summary')

        data = {'prompt': prompt, 
                'type':file_type,
                'use_sample': use_sample,
                'file_name': file_name,
                'sampling_rate':sampling_rate,
                'num_top_k': num_top_k,
                'conf_thres': conf_thres,
                'sim_thres': sim_thres,
                }
        
        df = pd.DataFrame({"name": [key.capitalize().replace('_', ' ') for key in data.keys()], 
                           "Input values": [str(v) for v in data.values()]})

        col2.dataframe(df, column_config={"name": "Configuration"}, use_container_width=True, hide_index=True)
        
        files = {'file': uploaded_file.getvalue()} if use_sample == 'No' else {'file': None}
        
        status_list = ["received parameters",
                        "Detecting people...",
                        "Calculating similarity...",
                        "Generating captions...",
                        "completed"]

        spinner_text = '서버에 요청 중...'
        
        with col2, st.spinner(spinner_text):
            response = requests.post(gpu_server_ip+'/predict_video', files=files, data=data, stream=True)
            res1, res2 = tee(response.iter_lines())

        for line in res1:
            if line:
                event_data = line.decode('utf-8')
                event_data = event_data.replace("data: ", "")

                import json
                event = json.loads(event_data)
                # event = eval(event_data)
                
                if 'error' in event:
                    spinner_text = event['error']
                    col2.error(spinner_text)                        
                        
                if 'status' in event and event['status'] != 'completed':
                        spinner_text = status_list[status_list.index(event['status']) + 1]  
                        with col2, st.spinner(f"상태: {spinner_text}"):
                            while True:
                                try:
                                    line_ = next(res2)
                                    if line_ and 'status' in str(line_):
                                        event_= eval(line_.decode('utf-8').replace("data: ", ""))
                                        if event_['status'] == status_list[status_list.index(spinner_text) + 1]:
                                            break
                                except:
                                    break

                if 'result' in event:
                    for i in range(0, num_top_k, 2):
                        _, col1, _, col2, _ = st.columns([0.2, 1, 0.2, 1, 0.2], gap='small')
                        column = [col1, col2]
                        for j in range(2):
                            index = i+j
                            if num_top_k > index:
                                img_data = event['result'][index]['image']
                                caption = event['result'][index].get('caption', '')
                                similarity = event['result'][index].get('similarity', '')
                                seconds = event['result'][index].get('seconds', '')

                                img_bytes = base64.b64decode(img_data)
                                img = Image.open(io.BytesIO(img_bytes))
                                
                                minutes = int(seconds // 60)
                                seconds = seconds % 60
                                
                                caption = f'Top{index+1} similarity: {similarity}, time: {minutes}분 {seconds:.2f}초 \
                                            \n{caption}'
                                
                                column[j].image(img, use_column_width=True)
                                column[j].write(caption)
                            
                if event.get('status') == 'completed':
                    st.success("예측 완료!")
                    break

    elif button and not prompt:
        col2.warning("Write your prompt.")
