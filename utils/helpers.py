import cv2

def create_video(frames, width, height, fps, path, name):
  video = cv2.VideoWriter(path + name + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

  for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

  video.release()


  