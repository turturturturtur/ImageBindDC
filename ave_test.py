import os
import json
import cv2
import imageio
import subprocess

# 文件和文件夹路径
txt_file = 'data/AVE_Dataset/testSet.txt'
videos_dir = 'data/AVE_Dataset/AVE'
audios_dir = 'data/test_data/ave/audios'
frames_dir = 'data/test_data/ave/frames'
output_json = 'data/test_data/ave_test.json'
category_dict_file = 'data/test_data/ave_category_dict.json'

os.makedirs(audios_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

category_dict = {}
next_category_id = 0
samples = []

with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('&')
        if len(parts) != 5:
            print(f"格式错误：{line}")
            continue
            
        category, video_id, quality, start_time, end_time = parts
        try:
            start_time = int(start_time)
            end_time = int(end_time)
        except ValueError:
            print(f"时间格式错误：{line}")
            continue

        if category not in category_dict:
            category_dict[category] = next_category_id
            next_category_id += 1
        label = category_dict[category]

        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"视频文件不存在：{video_path}")
            continue

        # 使用OpenCV获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频：{video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()  # 先释放，后续每个片段单独处理

        # 处理每个1秒片段
        for sec in range(start_time, end_time):
            if (sec + 1) > duration:
                print(f"跳过超出时长的片段：{video_id} [{sec}, {sec+1})")
                continue

            # 生成文件名
            audio_filename = f"{video_id}_{sec}.wav"
            audio_path = os.path.join(audios_dir, audio_filename)
            frame_filename = f"{video_id}_{sec}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            # 使用ffmpeg提取音频（精确切割）
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(sec),
                '-i', video_path,
                '-t', '1',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-hide_banner',
                '-loglevel', 'error',
                audio_path
            ]
            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            
            # 检查音频是否生成成功
            if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"音频提取失败：{video_id} {sec}，错误：{result.stderr.decode().strip()}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # 使用OpenCV提取中间帧
            cap = cv2.VideoCapture(video_path)
            mid_time = sec + 0.5
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"帧提取失败：{video_id} {mid_time}s")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # 转换颜色空间并保存
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageio.imwrite(frame_path, frame_rgb)

            samples.append({
                "audio": audio_path,
                "frame": frame_path,
                "label": label
            })

# 保存结果
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

with open(category_dict_file, 'w', encoding='utf-8') as f:
    json.dump(category_dict, f, indent=4, ensure_ascii=False)

print("处理完成，共生成样本数：", len(samples))