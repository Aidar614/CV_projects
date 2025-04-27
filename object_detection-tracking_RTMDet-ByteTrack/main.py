from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from mmdet.utils import register_all_modules
import supervision as sv
from visualiser import visualizer
from utils import (
    read_video,
    detect_frames,
    save_video
)


def main():
    config_file = 'mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py'
    checkpoint_file = 'rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth'
   
    register_all_modules()

    model = init_detector(config_file, checkpoint_file, device='cuda')
    tracker = sv.ByteTrack(track_activation_threshold=0.37, lost_track_buffer=120, minimum_matching_threshold=0.82,
    frame_rate =23, minimum_consecutive_frames=2)  
    
  

    frames, fps = read_video('Ho.mp4')
    results = detect_frames(frames, model, tracker)
    annotated_images = visualizer(frames, results)
    save_video(annotated_images, 'output_video0.37,120,0.82,2.mp4', fps)





if __name__ == "__main__":
    main()