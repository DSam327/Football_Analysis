import cv2
import sys
sys.path.append('../')
from utils import measure_dist, get_foot

class Speed:
    def __init__(self) -> None:
        self.frame_win=5
        self.fps=24

    def cal_speed(self, tracks):
        total_distance={}
        for obj, obj_track in tracks.items():
            if obj=='ball' or obj=='referee':
                continue

            for frame_num in range(0, len(obj_track), self.frame_win):
                last_frame=min(len(obj_track)-1, frame_num+self.frame_win)

                for track_id,_ in obj_track[frame_num].items():
                    if track_id not in obj_track[last_frame]:
                        continue

                    start=obj_track[frame_num][track_id]['position_transformed']
                    end=obj_track[last_frame][track_id]['position_transformed']

                    if start is None or end is None:
                        continue

                    dist= measure_dist(start, end)
                    time=(last_frame-frame_num)/self.fps
                    speed=dist/time
                    act_speed=speed*3.6

                    if obj not in total_distance:
                        total_distance[obj]= {}
                    
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0
                    
                    total_distance[obj][track_id] += dist

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        tracks[obj][frame_num_batch][track_id]['speed'] = act_speed
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames
