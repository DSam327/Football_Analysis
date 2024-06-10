# Football_Analysis
The project aims to analyze football games to detect players, balls and calculate various metrics like ball possession, and player movement. The detection and tracking of players, balls, and referees is done through YOLO model which is then fine-tuned with football images using Roboflow. Additionaly, The project uses KNN model to identify the team of the detected players their jersey colors. This helps with calculating ball possession. To ensure a hundred percent of detection of ball, we use interpolation to identify balls in non detected frames. To calculate the camera movement, optical flow is used. Finally, a speed and distance travelled metric is given to each player using perspective transformer along with a possession percentage of the ball for each team. 

![image](https://github.com/DSam327/Football_Analysis/assets/113661235/4ad8aa20-dedf-45b0-8157-12fa32973f20)



