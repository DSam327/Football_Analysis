from utils import read_video, write_video
from trackers import Tracker

def main():
    video_frames=read_video('./input_videos/08fd33_4.mp4')

    tracker=Tracker("./models/best.pt")

    tracks=tracker.object_tracks(video_frames, read_from_stub=True ,stub_path='./stubs/trackedplayers.pkl')

    output_video_frames=tracker.draw_annotation(video_frames, tracks)

    write_video(output_video_frames, './output_video/out.avi')

if __name__=='__main__':
    main()