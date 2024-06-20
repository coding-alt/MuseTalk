import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import load_all_model
from moviepy.editor import *
import shutil

import threading
import queue

import time
import gradio as gr
import uuid
import re

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None  

@torch.no_grad() 
class Avatar:
    def __init__(self, avatar_id, batch_size):
        self.avatar_id = avatar_id
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_dir = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.work_temp_dir = f"{self.avatar_path}/{str(uuid.uuid4())}"
        self.batch_size = batch_size
        self.idx = 0
        self.init()
        
    def init(self):
        if not os.path.exists(self.latents_out_path):
            raise Exception(f"latents({self.latents_out_path}) not exists!")
                
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        if not os.path.exists(self.full_imgs_path):
            raise Exception(f"full_imgs({self.full_imgs_path}) not exists!")
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)

        if not os.path.exists(self.mask_coords_path):
            raise Exception(f"mask({self.mask_coords_path}) not exists!")
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
        
    def process_frames(self, 
                       res_frame_queue,
                       video_len):
        print(video_len)
        while True:
            if self.idx>=video_len-1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
      
            bbox = self.coord_list_cycle[self.idx%(len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx%(len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx%(len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx%(len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            cv2.imwrite(f"{self.work_temp_dir}/{str(self.idx).zfill(8)}.png",combine_frame)
            self.idx = self.idx + 1

    def inference(self, 
                  audio_path: str, 
                  infer_video_path: str = None, 
                  fps: int = 30):
        os.makedirs(self.work_temp_dir, exist_ok =True)  
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue()
        self.idx = 0
        # # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle, 
                      self.batch_size)
        start_time = time.time()
        
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch, 
                                      timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()
        
        print('Total process time of {} frames including saving images = {}s'.format(
                    video_num,
                    time.time()-start_time))
        
        if infer_video_path is None:
            infer_video_path = os.path.join(self.video_out_dir, str(uuid.uuid4()) + '.mp4')

        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        print('Start combining images...')
        tmp_img_save_path = self.work_temp_dir
        files = [file for file in os.listdir(tmp_img_save_path) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))
        img_list = [os.path.join(tmp_img_save_path, file) for file in files]

        tmp_video = os.path.join(self.avatar_path, str(uuid.uuid4()) + '.mp4')
        print(f"Merge video...")

        # 创建图像序列剪辑
        video_clip = ImageSequenceClip(img_list, fps=fps)
        video_clip.write_videofile(tmp_video, fps=fps, codec='libx264', audio=False)

        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        print('Video is ready to be saved...')
        video_clip.write_videofile(infer_video_path, codec='libx264', audio_codec='aac', fps=fps)

        audio_clip.close()
        video_clip.close()

        # 清理临时目录和文件
        os.remove(tmp_video)
        shutil.rmtree(self.work_temp_dir)

        print(f"result is save to {infer_video_path}")
        return infer_video_path

def inference(avatar_id, audio_path, batch_size, fps):
    if avatar_id is None or audio_path is None:
        raise gr.Error("请确保上传音频文件，并选择推理模型！")
    
    print("Inference use model:", avatar_id)
    avatar = Avatar(avatar_id, batch_size)
    return avatar.inference(audio_path, fps=fps)


avatar_path = f"./results/avatars/"
def get_avatar_list():
    return os.listdir(avatar_path)

def refresh_avatar_list():
    return gr.Radio(choices=get_avatar_list(), label="模型列表")

css = """footer {visibility: hidden}"""
app = gr.Blocks(title="MuseTalk数字人推理", css=css, theme="Kasien/ali_theme_custom")

with app:
    gr.Markdown("# <center>🎡 - MuseTalk数字人推理</center>")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem('推理音频'):
                    audio_path = gr.Audio(sources="upload", type="filepath", label="支持mp3、wav格式")
            
            with gr.Tabs():
                with gr.TabItem('设置'):
                    avatar_id = gr.Radio(choices=get_avatar_list(), label="模型列表")
                    batch_size = gr.Slider(1, 128, 8, step=1, label="batch_size", info="加速推理选项，根据GPU实际情况调整")
                    fps = gr.Number(value=30, label="fps", info="视频每秒显示的帧数")

                    btn_inference = gr.Button("一键推理")
                    btn_refresh_model = gr.Button("刷新模型列表")

        with gr.Column():
            output_video = gr.Video(label="推理结果")

    btn_inference.click(inference, [avatar_id, audio_path, batch_size, fps], output_video)
    btn_refresh_model.click(refresh_avatar_list, None, avatar_id)

app.launch(server_name='0.0.0.0')