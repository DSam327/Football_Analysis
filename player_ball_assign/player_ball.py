import sys
sys.path.append('../')
from utils import get_center, measure_dist

class PLayerball:
    def __init__(self):
        self.max_dist=60

    def assign_ball(self, players, ball):
        ball_pos= get_center(ball)
        min_dist=1000000
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox=player['bbox']

            dist_left=measure_dist((player_bbox[0], player_bbox[-1]), ball_pos)
            dist_right=measure_dist((player_bbox[2], player_bbox[-1]), ball_pos)

            dist=min(dist_left, dist_right)
            if dist<self.max_dist:
                if dist<min_dist:
                    min_dist=dist
                    assigned_player=player_id
        
        return assigned_player
