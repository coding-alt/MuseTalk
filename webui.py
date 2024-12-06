import streamlit as st
import os
import uuid
from train import Train
from inference import Inference

# 定义两个方法
def train(avatar_id: str, video_path: str, bbox_shift: int = 0):
    try:
        musetalk = Train(
            avatar_id = avatar_id, 
            video_path = video_path, 
            bbox_shift = bbox_shift
        )
        state = musetalk.run()
        if state:
            os.remove(video_path)
            st.success(f"训练成功！可以开始推理了...")
        else:
            st.error(f"训练失败！")
    except Exception as e:
        st.error(f"训练失败！")

def inference(avatar_id: str, audio_path: str, batch_size: int = 8, fps: int = 25, enhance = True, save_path: str = None):
    try:
        musetalk = Inference(avatar_id, batch_size)
        state = musetalk.run(audio_path, infer_video_path=save_path, fps=fps, enhance=enhance)
        if state:
            st.video(save_path)
        else:
            st.error(f"推理失败！")
    except Exception as e:
        st.error(f"推理失败！")
    finally:
        os.remove(audio_path)

def train_page():
    st.header("训练", divider='rainbow')
    avatar_id = st.text_input("数字人ID")
    video_file = st.file_uploader("上传训练素材 (mp4文件)", type=["mp4"])
    bbox_shift = st.slider("bbox_shift", min_value=-25, max_value=25, value=5)
    
    if st.button("开始训练"):
        if avatar_id and video_file:
            video_path = f"./{str(uuid.uuid4())}.mp4"  # 保存上传的视频文件
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            with st.spinner('训练中，请稍候...'):
                train(avatar_id, video_path, bbox_shift)
        else:
            st.error("请提供完整的训练参数")

def get_avatar_list():
    avatar_path = f"./results/avatars/"
    return os.listdir(avatar_path)

def inference_page():
    st.header("推理", divider='rainbow')
    avatar_id = st.radio("数字人ID", get_avatar_list(), horizontal=True)
    audio_file = st.file_uploader("上传音频文件", type=["mp3", "wav", "flac"])
    batch_size = st.slider("batch_size", min_value=1, max_value=16, value=8)
    fps = int(st.text_input("fps", value=25))
    enhance = st.checkbox("唇部高清化", value=True)
    save_path = f"./results/avatars/{avatar_id}/vid_output/{str(uuid.uuid4())}.mp4"
    
    if st.button("开始推理"):
        if avatar_id and audio_file:
            audio_path = f"./{str(uuid.uuid4())}.{audio_file.name[-3:]}"  # 保存上传的音频文件
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            with st.spinner('推理中，请稍候...'):
                inference(avatar_id, audio_path, batch_size, fps, enhance, save_path)
        else:
            st.error("请提供完整的推理参数")

pg = st.navigation([
    st.Page(train_page, title="训练", icon=":material/video_call:"),
    st.Page(inference_page, title="推理", icon=":material/data_saver_on:"),
])
pg.run()
