from utils import read_video, write_video
from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assign import PLayerball
from camera_movement import camera_move
from view_transform import ViewTransformer
from speed_dist import Speed

def main():
    video_frames=read_video('./input_videos/08fd33_4.mp4')

    tracker=Tracker("./models/best.pt")

    tracks=tracker.object_tracks(video_frames, read_from_stub=True ,stub_path='./stubs/trackedplayers.pkl')
    tracker.add_position_to_track(tracks)
    camera=camera_move(video_frames[0])
    camera_move_per_frame=camera.get_camera_movement(video_frames, read_from_stub=True, stub_path='./stubs/camera_move.pkl')
    camera.adjust_position(tracks, camera_move_per_frame)
    
    tr=ViewTransformer()
    tr.transformed_position_to_track(tracks)
    
    tracks['ball']=tracker.interpolate_ball(tracks['ball'])

    speed_cal=Speed()
    speed_cal.cal_speed(tracks)


    team_assign=TeamAssigner()
    team_assign.assign_team_color(video_frames[0], tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team=team_assign.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team']=team
            tracks['player'][frame_num][player_id]['team_color']=team_assign.team_colors[team]

    team_control=[]
    assign_ball=PLayerball()
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox=tracks['ball'][frame_num][1]['bbox']
        assigned_player= assign_ball.assign_ball(player_track, ball_bbox)

        if assigned_player>-1:
            tracks['player'][frame_num][assigned_player]['has_ball']=True
            team_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            if frame_num<=1:
                team_control.append(1)
            else:
                team_control.append(team_control[-1])

    team_control=np.array(team_control)
    output_video_frames=tracker.draw_annotation(video_frames, tracks, team_control)
    draw_camera=camera.draw_camera_frame(output_video_frames, camera_move_per_frame)
    speed_cal.draw_speed_and_distance(draw_camera, tracks)
    write_video(draw_camera, './output_video/out.avi')

if __name__=='__main__':
    main()