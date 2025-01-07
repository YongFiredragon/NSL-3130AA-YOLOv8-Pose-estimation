from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from lidar_test2 import NSL3130AA

def __main__ ():
    # Load the YOLOv8 model
    model = YOLO('yolov8s.pt')
    # Setting for GPU ACC
    device = torch.device("mps")
    model.to(device)
    # Open the video file
    video_path = "./sample.mp4"
    print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
    print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")

    lidar = NSL3130AA(ipaddr="192.168.241.254", port=50660)
    lidar.connect()
    lidar.startCaptureCommand()
    # cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    # while cap.isOpened():
    while True:
        # Read a frame from the video
        # success, frame = cap.read()
        success, frame = lidar.read()
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=0, conf=0.4)

            # Get the boxes and track IDs
            # boxes = results[0].boxes.xywh.cpu()
            # track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes  # Boxes 객체
            boxes_data = boxes.data  # shape (N,7) (트래킹 시)
            if boxes.is_track:  # 트래킹 모드
                track_ids = boxes_data[:, 6].int().cpu().tolist()
            else:
                track_ids = [None] * boxes_data.shape[0]  # 트랙 ID 없음

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                print(f"Box data: {box}")  # 디버깅용
                x, y, w, h = box.xywh[0]
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                print(f"%s", track_id)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    # cap.release()
    lidar.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()

