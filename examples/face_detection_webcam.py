# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# face detection webcam example
# usage: python face_detection_webcam.py 

# import necessary packages
import cvlib as cv
import cv2
import time
import threading
import numpy as np
from calculate_rotation import *
# open webcam
webcam = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 30
# out = cv2.VideoWriter('save_video/얼굴 탐지.avi', fourcc, fps, (640, 480))

end = 0
w,h=640, 480
# mid_x,mid_y=w/2, h/2
point_flag = False
lock = threading.Lock()
sigmaX,sigmaA=0.001,5
dt = 1/30
kalman_flag = False
cpx, cpy = w/2, h/2
startX, startY,endX,endY=0,0,0,0
tracking_flag = False
predict_x, predict_y = 0,0
current_prediction=0

F=[ [1, dt, 0.5*dt**2, 0, 0, 0],
      [0, 1, dt, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5*dt**2],
      [0, 0, 0, 0, 1, dt],
      [0, 0, 0, 0, 0, 1]]
H=[ [1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0]]
Q=[ [dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
      [dt**3/2, dt**2, dt, 0, 0, 0],
      [dt**2/2, dt, 1, 0, 0, 0],
      [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
      [0, 0, 0, dt**3/2,dt**2, dt],
      [0, 0, 0, dt**2/2, dt, 1]]
R= [[1,0],[0,1]]

kalman = cv2.KalmanFilter(6, 2)   #  state [ x, vx, ax, y, vy, ay] (상태벡터의 차원 : 6, 측정벡터의 차원 : 2)
kalman.transitionMatrix = np.array(F, np.float32)
kalman.measurementMatrix = np.array(H, np.float32) # System measurement matrix
kalman.processNoiseCov = np.array(Q, np.float32)*sigmaA**2 # System process noise covariance
kalman.measurementNoiseCov = np.array(R, np.float32) * sigmaX ** 2

cx, cy = w/2, h/2

# initial value
# 예측상태
kalman.statePre = np.array(
    [[cx], [0], [0], [cy], [0], [0]], np.float32)
# 정정상태
kalman.statePost = np.array(
    [[cx], [0], [0], [cy],[0],[0]], np.float32)



def goto_human():
    global point_flag,mid_x,mid_y,cpx, cpy,nx,ny,predict_x, predict_y,current_prediction

    while True:
        lock.acquire()
        if point_flag:
            print('working')
            print(cpx, cpy)
           # print(kalman.errorCovPost)

        else:
            print('not working')
            # print(predict_x, predict_y)

        lock.release()


#
# if not webcam.isOpened():
#     print("Could not open webcam")
#     exit()
#
if __name__ == '__main__':
    # loop through frames
    fps = 30
    pre_time = time.time()
    fps_data = []

    human = threading.Thread(target=goto_human)
    human.daemon = True  # 프로그램 종료시 즉시 종료.
    human.start()


    if not webcam.isOpened():
        print("Could not open webcam")
        exit()


    while True:
        if webcam.isOpened():
            # read frame from webcam
            status, frame = webcam.read()


            if not status:
                print("Could not read frame")
                exit()

            now_time = time.time()
            delta = now_time - pre_time
            fps = 1/delta
            if fps > 30:
                fps = 30
            fps_data.append(fps)
            # apply face detection
            start = time.time()
            face, confidence = cv.detect_face(frame)

            # print(face)
            # print(confidence)

            # loop through detected faces

            for idx, f in enumerate(face):
                point_flag = True
                startX, startY = f[0], f[1]
                endX, endY = f[2], f[3]
                cx, cy = int((startX+endX)/2), int((startY+endY)/2)
                 # for drawing figure in face
                frame = cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
                frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)

            current_prediction = kalman.predict() # start kalman filter <predict>
            cpx, cpy = int(current_prediction[0]), int(current_prediction[3])
            current_measurement = np.array([cx, cy], np.float32)
            kalman.correct(current_measurement)
            frame = cv2.circle(frame, (cpx, cpy), 8, (0, 0, 255), -1)

            #
            # # kalman filter <measurement>
            # current_measurement = np.array([[mid_x], [mid_y]], np.float32)
            # estimate = np.int0(kalman.correct(current_measurement))
            # predict_x, predict_y = estimate[0][0],estimate[3][0]
            # # print(kalman.gain)
            # # frame = cv2.circle(frame, (predict_x, predict_y), 8, (255, 0, 0), -1)




            end = time.time()-start

                # draw rectangle over face

            pre_time = now_time
            # display output
            cv2.putText(frame, "FPS: %3.1f" % (fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Real-time face detection", frame)
            # out.write(frame)

            # press "Q" to stop
            if cv2.waitKey(1) == 27:
                break


    # release resources
    # out.release()
    webcam.release()
    cv2.destroyAllWindows()
