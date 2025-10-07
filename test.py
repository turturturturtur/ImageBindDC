import os
import subprocess

video_id = 'XGVkq0Kl2aw'
sec = 1

videos_dir = 'data/AVE_Dataset/AVE'
audios_dir = 'data/test_data/ave/audios'
os.makedirs(audios_dir, exist_ok=True)

video_path = os.path.join(videos_dir, f"{video_id}.mp4")
audio_filename = f"{video_id}_{sec}.wav"
audio_path = os.path.join(audios_dir, audio_filename)

# ffmpeg命令，截取第1秒长度的音频
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
else:
    print(f"生成成功：{audio_path}")
