from ultralytics import YOLO
import supervision as sv
import os
import cv2
import pickle
import sys
import numpy as np
sys.path.append('../')
from utils import get_width, get_center

class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def detect_frames(self, frames):
        detections=[]
        batch_size=20
        for i in range(0, len(frames), batch_size):
            detection=self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections+=detection
        return detections
    
    def object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub==True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks=pickle.load(f)
            return tracks
        
        detections=self.detect_frames(frames)


        tracks= {
            'player' :[],
            'referee' : [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names=detection.names
            cls_inv={v:k for k,v in cls_names.items()}

            detection_supervision=sv.Detections.from_ultralytics(detection)

            for obj_ind, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id]=="goalkeeper":
                    detection_supervision.class_id[obj_ind]=cls_inv["player"]
            
            detection_track=self.tracker.update_with_detections(detection_supervision)

            tracks['player'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})

            for frame_detect in detection_track:
                bbox=frame_detect[0].tolist()
                cls_id= frame_detect[3]
                track_id=frame_detect[4]

                if cls_id==cls_inv['player']:
                    tracks['player'][frame_num][track_id]={"bbox":bbox}

                if cls_id==cls_inv['referee']:
                    tracks['referee'][frame_num][track_id]={"bbox":bbox}
            
            for frame_detect in detection_supervision:
                bbox=frame_detect[0].tolist()
                cls_id=frame_detect[3]

                if cls_id==cls_inv['ball']:
                    tracks['ball'][frame_num][1]={"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_triangle(self, frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)
        return frame

    def draw_ellipse(self, frame ,bbox, bgr, id=None):
        y2=int(bbox[3])
        x_center=get_center(bbox)[0]
        width=get_width(bbox)
        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35*width)), 0.0, startAngle=-45, endAngle=235, color=bgr, thickness=2, lineType=cv2.LINE_4)

        rect_h=20
        rect_w=40
        x1_rect=x_center-rect_w//2
        x2_rect=x_center+rect_w//2
        y1_rect=y2-rect_h//2+15
        y2_rect=y2+rect_h//2+15

        if id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), bgr, cv2.FILLED)

            x1_text=x1_rect+12
            if id>99:
                x1_text-=10

            cv2.putText(frame, str(id), (x1_text, y1_rect+15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255))
        return frame

        

    def draw_annotation(self, video_frames, tracks):
        output=[]
        for frame_num, frame in enumerate(video_frames):
            frame=frame.copy()
            players_dict=tracks['player'][frame_num]
            referees_dict=tracks['referee'][frame_num]
            ball_dict=tracks['ball'][frame_num]

            for track_id, player in players_dict.items():
                frame=self.draw_ellipse(frame ,player['bbox'], (0,0,255),id=track_id)
            
            for track_id, referee in referees_dict.items():
                frame=self.draw_ellipse(frame, referee['bbox'], (0,255,255))

            for ball_id, ball in ball_dict.items():
                frame=self.draw_triangle(frame, ball['bbox'], (255,255,255))
            
            
            output.append(frame)
        
        return output




        

