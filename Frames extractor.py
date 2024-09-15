import cv2
import yt_dlp
import os
import colorsys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def list_to_string(lst, num_digits=2):
    rounded_strings = [f"{int(round(num)):0{num_digits}d}" for num in lst]
    return ''.join(rounded_strings)

def rgb_to_hsb(r, g, b):
    # Ensure r, g, b are floats
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
    h = h * 98 + 1
    s = s * 98 + 1
    v = v * 98 + 1
    return h, s, v

def process_frame(frame):
    # Resize the frame to 50x50 pixels
    frame = cv2.resize(frame, (50, 50))
    
    # Convert to numpy array
    frame = np.array(frame)
    b, g, r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    
    # Vectorized conversion to HSB
    h, s, v = rgb_to_hsb(r, g, b)
    
    # Flatten arrays
    hue_str = list_to_string(h.flatten())
    saturation_str = list_to_string(s.flatten())
    brightness_str = list_to_string(v.flatten())
    
    return hue_str, saturation_str, brightness_str

def download_youtube_video(url):
    try:
        video_path = 'downloaded_video.mp4'
        # Remove existing file if it exists
        if os.path.exists(video_path):
            os.remove(video_path)
        
        ydl_opts = {
            'format': 'bestvideo',
            'outtmpl': video_path,
            'noplaylist': True,
            'merge_output_format': None,  # Disable merging
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return video_path
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None

def process_youtube_video(url):
    video_path = download_youtube_video(url)
    
    if video_path is None:
        print("Failed to download or process the video.")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        frame_count = 0
        hsb_data = []
        
        def process_frame_task(frame):
            nonlocal frame_count
            frame_count += 1
            hue_str, saturation_str, brightness_str = process_frame(frame)
            return f"Frame {frame_count}: {hue_str}|{saturation_str}|{brightness_str}|\n"
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                futures.append(executor.submit(process_frame_task, frame))
                
            # Write results to file
            with open("HSB_data.txt", "w") as file:
                for future in futures:
                    file.write(future.result())
            
            print(f"Total frames processed: {frame_count}")
        
        cap.release()
    
    except Exception as e:
        print(f"An error occurred while processing the video: {e}")

def main():
    url = input("Please enter the YouTube video URL: ")
    process_youtube_video(url)

if __name__ == "__main__":
    main()
