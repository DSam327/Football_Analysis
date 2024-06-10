import numpy as np
import cv2

class ViewTransformer:
    def __init__(self) -> None:
        width=68
        length=23.32
        self.pixel_vertices=np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        self.target_vertices = np.array([
            [0,width],
            [0, 0],
            [length, 0],
            [length, width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_pnt(self, position):
        p = (int(position[0]),int(position[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = position.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)

    def transformed_position_to_track(self, tracks):
        for object,object_tracks in tracks.items():
            for frame_num, frame_track in enumerate(object_tracks):
                for track_id, player_track in frame_track.items():
                    position=player_track['position_adj']
                    position=np.array(position)
                    position_transformed = self.transform_pnt(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed

    