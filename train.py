import argparse
import os
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import json
from musetalk.utils.preprocessing import get_landmark_and_bbox,coord_placeholder
from musetalk.utils.blending import get_image_prepare_material
from musetalk.utils.utils import load_all_model
import shutil
import queue
import time

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

def video2imgs(vid_path, save_path, ext = '.png', cut_frame = 10000000):
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
        os.makedirs(path, exist_ok = True)
@torch.no_grad() 
class Train:
    def __init__(self, avatar_id: str, video_path: str, bbox_shift: int = 5, batch_size: int = 4):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_dir = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id":avatar_id,
            "video_path":video_path,
            "bbox_shift":bbox_shift
        }
        self.batch_size = batch_size
        self.idx = 0
        
    def run(self):
        if os.path.exists(self.avatar_path):
            shutil.rmtree(self.avatar_path)
            print(f"avatar({self.avatar_id}) exists, remove it!")

        print("*********************************")
        print(f"  creating avator: {self.avatar_id}")
        print("*********************************")
        osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_dir,self.mask_out_path])
        try:
            self.prepare_material()
            return True
        except:
            return False
    
    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
            
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext = '.png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1]=="png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient 
        coord_placeholder = (0.0,0.0,0.0,0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i,frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
            face_box = self.coord_list_cycle[i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
            
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
            
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

        torch.cuda.empty_cache()
        
        print("Training successful!") 
        
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
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue

            self.idx = self.idx + 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--avatar_id", 
                        type=str,
                        help="avatar id",
    )
    parser.add_argument("--video_path", 
                        type=str,
                        help="video path",
    )
    parser.add_argument("--bbox_shift", 
                        type=int, 
                        default=5,
                        help="bbox shift value",
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4,
                        help="batch size",
    )

    args = parser.parse_args()

    if args.avatar_id is None or args.video_path is None:
        raise ValueError("Please provide --avatar_id, --video_path")
    
    print("Train model: ", args.avatar_id)
    start = int(time.time())
    musetalk = Train(
        avatar_id = args.avatar_id, 
        video_path = args.video_path, 
        bbox_shift = args.bbox_shift, 
        batch_size = args.batch_size
    )
    state = musetalk.run()
    endtime = int(time.time())
    print(f"Training time: {endtime-start}s")