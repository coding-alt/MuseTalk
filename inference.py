import os
import re
import cv2
import time
import uuid
import glob
import copy
import torch
import queue
import pickle
import shutil
import argparse
import threading
import subprocess
import numpy as np
from tqdm import tqdm
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import load_all_model
from utils.gfpgan_wrapper import GfpganEnhancer
from moviepy.editor import ImageSequenceClip, AudioFileClip

@torch.no_grad() 
class Inference:
    # Class-level variables to hold the model and state
    model_loaded = False
    audio_processor = None
    vae = None
    unet = None
    pe = None
    gfpgan_enhancer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)

    frame_list_cache = {}
    coord_list_cache = {}
    mask_list_cache = {}
    mask_coords_cache = {}
    latent_list_cache = {}

    def __init__(self, avatar_id: str, batch_size: int = 8):
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
        self.load_model()

    @classmethod
    def load_model(cls):
        if not cls.model_loaded:
            print("Loading models...")
            cls.audio_processor, cls.vae, cls.unet, cls.pe = load_all_model()
            cls.pe = cls.pe.half()
            cls.vae.vae = cls.vae.vae.half()
            cls.unet.model = cls.unet.model.half()
            cls.gfpgan_enhancer = GfpganEnhancer()
            cls.model_loaded = True
            print("Models loaded successfully.")
        
    def init(self):
        if self.avatar_id not in self.latent_list_cache:
            if not os.path.exists(self.latents_out_path):
                raise Exception(f"latents({self.latents_out_path}) not exists!")
            self.latent_list_cache[self.avatar_id] = torch.load(self.latents_out_path)
        self.input_latent_list_cycle = self.latent_list_cache[self.avatar_id]

        if self.avatar_id not in self.coord_list_cache:
            with open(self.coords_path, 'rb') as f:
                self.coord_list_cache[self.avatar_id] = pickle.load(f)
        self.coord_list_cycle = self.coord_list_cache[self.avatar_id]

        if self.avatar_id not in self.frame_list_cache:
            if not os.path.exists(self.full_imgs_path):
                raise Exception(f"full_imgs({self.full_imgs_path}) not exists!")
            input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cache[self.avatar_id] = read_imgs(input_img_list)
        self.frame_list_cycle = self.frame_list_cache[self.avatar_id]

        if self.avatar_id not in self.mask_coords_cache:
            if not os.path.exists(self.mask_coords_path):
                raise Exception(f"mask({self.mask_coords_path}) not exists!")
            with open(self.mask_coords_path, 'rb') as f:
                self.mask_coords_cache[self.avatar_id] = pickle.load(f)
        self.mask_coords_list_cycle = self.mask_coords_cache[self.avatar_id]

        if self.avatar_id not in self.mask_list_cache:
            input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cache[self.avatar_id] = read_imgs(input_mask_list)
        self.mask_list_cycle = self.mask_list_cache[self.avatar_id]
        
    def process_frames(self, res_frame_queue, video_len, enhance):
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
            if enhance:
                res_frame = self.gfpgan_enhancer.enhance(res_frame)
            
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            cv2.imwrite(f"{self.work_temp_dir}/{str(self.idx).zfill(8)}.png",combine_frame)
            self.idx = self.idx + 1

    def run(self, 
                  audio_path: str, 
                  infer_video_path: str = None, 
                  fps: int = 25,
                  enhance: bool = False):
        os.makedirs(self.work_temp_dir, exist_ok =True)  
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue()
        self.idx = 0
        # # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, enhance))
        process_thread.start()
        
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                         dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, 
                                      self.timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
                
        process_thread.join()
        torch.cuda.empty_cache()
        
        print(f"Total process time of {video_num} frames including saving images = {(time.time() - start_time):.2f} seconds")
        
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

        print(f"Merge video...")
        start_time = time.time()
        
        try:
            # 创建图像序列剪辑
            video_clip = ImageSequenceClip(img_list, fps=fps)

            audio_clip = AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            print('Video is ready to be saved...')
            video_clip.write_videofile(infer_video_path, codec='libx264', audio_codec='aac', fps=fps)

            audio_clip.close()
            video_clip.close()

            print(f"Inference successful, result is save to {infer_video_path}, Merge video costs {(time.time() - start_time):.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Inference failed, reason: {e}")
            return False
        finally:
            if os.path.exists(self.work_temp_dir):
                shutil.rmtree(self.work_temp_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--avatar_id", 
                        type=str,
                        help="avatar id",
    )
    parser.add_argument("--audio_path", 
                        type=str,
                        help="audio path",
    )
    parser.add_argument("--save_path", 
                        type=str,
                        help="inference result save path",
    )
    parser.add_argument("--fps",
                        type=int,
                        default=30,
                        help="fps",
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4,
                        help="batch size",
    )

    args = parser.parse_args()

    if args.avatar_id is None or args.audio_path is None:
        raise ValueError("Please provide --avatar_id, --audio_path")
    
    print("Inferencing use model:", args.avatar_id)
    start = int(time.time())

    musetalk = Inference(args.avatar_id, args.batch_size)
    state = musetalk.run(args.audio_path, infer_video_path=args.save_path, fps=args.fps)

    endtime = int(time.time())
    print(f"Inferencing use time: {endtime-start}s")
