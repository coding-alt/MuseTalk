import os
import cv2
import uuid
import re
import numpy as np
from gfpgan import GFPGANer
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

class VideoEnhancer:
    def __init__(self):
        # 加载模型
        CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CURRENT_SCRIPT_PATH, '../gfpgan/weights', 'GFPGANv1.3.pth')
        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None)
        self.restorer = restorer

    def enhance(self, input_img):
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)

        return restored_img

    def video_enhance(self, video_path):
        # 生成临时文件路径
        temp_dir = os.path.dirname(video_path)
        temp_file_name = str(uuid.uuid4())
        temp_img_dir = os.path.join(temp_dir, temp_file_name)
        os.makedirs(temp_img_dir, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)

        # 获取输入视频文件的基本信息
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps

        frame_count = 0

        # 循环读取视频中的每一帧
        while True:
            # 读取一帧视频
            success, image = video_capture.read()
            # 判断视频是否已经读取完毕
            if not success:
                break
            frame = self.enhance(image)
            frame_path = os.path.join(temp_img_dir, f"{frame_count:08d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        # 关闭视频文件
        video_capture.release()

        # 将图片合成视频
        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        print('Start combining images...')
        files = [file for file in os.listdir(temp_img_dir) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))
        img_list = [os.path.join(temp_img_dir, file) for file in files]

        tmp_video = os.path.join(temp_dir, f"{temp_file_name}.mp4")
        video_clip = ImageSequenceClip(img_list, fps=fps)
        video_clip.write_videofile(tmp_video, fps=fps, codec='libx264', audio=False)

        # 使用moviepy将源视频的音频拷贝过去合成新视频文件
        temp_video_clip = VideoFileClip(tmp_video)
        audio_clip = AudioFileClip(video_path)
        final_clip = temp_video_clip.set_audio(audio_clip)

        # 生成最终输出文件路径
        base_name, ext = os.path.splitext(os.path.basename(video_path))
        final_output_path = os.path.join(temp_dir, f"{base_name}_enhanced.mp4")
        final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

        # 删除中间的临时文件
        for file in files:
            os.remove(os.path.join(temp_img_dir, file))
        os.rmdir(temp_img_dir)
        os.remove(tmp_video)

        return final_output_path

if __name__ == "__main__":
    video_enhancer = VideoEnhancer()
    video_path = "path_to_your_video.mp4"
    enhanced_video_path = video_enhancer.video_enhance(video_path)
    print(f"Enhanced video saved to: {enhanced_video_path}")