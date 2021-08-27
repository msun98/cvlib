# import necessary packages
import cvlib as cv
import threading
from calculate_rotation import *
from initial import *
from ptz_api import *
import time
import numpy as np

cpx_old = True
end = 0
w_calibration, h_calibration = 1527, 833
mid_x, mid_y = w_calibration/2, h_calibration/2
point_flag = False
lock = threading.Lock()
sigmaX, sigmaA = 0.01, 0.015
dt = 1/20
k_x, k_y = w_calibration/2, h_calibration/2
startX, startY, endX, endY = 0, 0, 0, 0
cx, cy = 0, 0
cpx, cpy = 0, 0
predict_x, predict_y = 0, 0


F = [ [1, dt, 0.5*dt**2, 0, 0, 0],
      [0, 1, dt, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5*dt**2],
      [0, 0, 0, 0, 1, dt],
      [0, 0, 0, 0, 0, 1]]

H = [ [1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0]]

Q = [ [dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
      [dt**3/2, dt**2, dt, 0, 0, 0],
      [dt**2/2, dt, 1, 0, 0, 0],
      [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
      [0, 0, 0, dt**3/2,dt**2, dt],
      [0, 0, 0, dt**2/2, dt, 1]]

R = [[1,0],[0,1]]


kalman = cv2.KalmanFilter(6, 2)   #  state [ x, vx, ax, y, vy, ay]
kalman.transitionMatrix = np.array(F, np.float32)
kalman.measurementMatrix = np.array(H, np.float32) # System measurement matrix
kalman.processNoiseCov = np.array(Q, np.float32)*sigmaA**2 # System process noise covariance
kalman.measurementNoiseCov = np.array(R, np.float32) * sigmaX**2

cx, cy = w_calibration/2, h_calibration/2

kalman.statePre = np.array([[cx], [0], [0], [cy],[0],[0]], np.float32)
kalman.statePost = np.array([[cx], [0], [0], [cy],[0],[0]], np.float32)

def goto_human():
    global point_flag,predict_x,predict_y,cpx, cpy
    angle_of_x_old, angle_of_y_old = 0, 0
    pp, tp = 0, 0
    predict_x_old = 0
    start = time.time()

    while True:
        lock.acquire()
        if point_flag:
            print('working')

            if np.abs(predict_x-predict_x_old) > 500:
                moveTo(int(pan*100),int(tilt*100),0,0)

            else:
                # view2sphere(cpx, cpy, 0)
                # pan, tilt = camera2world()
                pan, tilt = np.rad2deg(calculate_alpha(cpx,0)),np.rad2deg(calculate_beta(cpy,0))
                # stop('right')
                # stop('up')
                # TO CALCULATE OF MOTOR SPEED
                end = 0.01
                # print(np.abs(pan-angle_of_x_old))
                angular_speed_x, angular_speed_y = round((np.abs(pan-angle_of_x_old)/end)*0.2,3),\
                                                   round((np.abs(tilt-angle_of_y_old)/end)*0.2,3) # 각속도 계산
                #
                pp, tp = int(0.825 * np.abs(angular_speed_x) + 0.127), int(0.825 * np.abs(angular_speed_y) + 0.127)

                # stan_by = time.time()-start
                # if stan_by > 2:
                print('motor')
                # if pp > 30:
                #     pp = 30
                # if tp > 10:
                #     tp = 30
                # moveTo(int(pan*100), int(tilt*100), int(pp/5), int(tp/20))

                if pan < 0:
                    if tilt > 0:
                        move_pan_tilt('right', 'up', pp, int(tp/20))
                    else:
                        move_pan_tilt('right', 'down', pp, int(tp/20))

                elif pan > 0:
                    if tilt > 0:
                        move_pan_tilt('left', 'up', pp, int(tp/20))
                    else:
                        move_pan_tilt('left', 'down', pp, int(tp/20))
                angle_of_x_old, angle_of_y_old = pan, tilt

        lock.release()


if __name__ == '__main__':

    vcap = initialize() # 화면을 받아옴.
    on_screen_display()
    time.sleep(0.3)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fps = 25
    pre_time = time.time()
    fps_data = []
    # out = cv2.VideoWriter('save_video/얼굴 탐지.avi', fourcc, fps, (1527, 833))

    # -------------------------처음 이미지 받아오기---------------------------------------------
    ret_val, image = vcap.read()
    video = calibration(image, w_calibration, h_calibration)
    cv2.namedWindow('Real-time face detection')
    # -------------------------------------------------------------------------------------

    initial_position = get_position()
    if initial_position != (0, 0, 0):
        goto_origin(pp,ps,tp,ts)
        time.sleep(0.3)
        
    human = threading.Thread(target=goto_human)
    human.daemon = True  # 프로그램 종료시 즉시 종료.
    human.start()

    # loop through frames
    while True:
        if vcap.isOpened():
            status, image = vcap.read()
            if not status:
                break

            now_time = time.time()
            delta = now_time - pre_time
            fps = 1/delta
            if fps > 30:
                fps = 30
            fps_data.append(fps)

            # apply face detection
            video = calibration(image, w_calibration, h_calibration)
            start = time.time()
            face, confidence = cv.detect_face(video)

            # loop through detected faces
            for idx, f in enumerate(face):
                point_flag = True
                startX, startY = f[0], f[1]
                endX, endY = f[2], f[3]
                cx, cy = int((startX+endX)/2), int((startY+endY)/2)

                # for drawing figure in face
                video = cv2.circle(video, (cx, cy), 8, (255, 0, 0), -1)
                # video = cv2.rectangle(video, (startX, startY), (endX, endY), (0,255,0), 2)

            current_prediction = kalman.predict() # start kalman filter <predict>
            cpx, cpy = int(current_prediction[0]), int(current_prediction[3])
            current_measurement = np.array([cx, cy], np.float32)
            kalman.correct(current_measurement)
            frame = cv2.circle(video, (cpx, cpy), 8, (0, 0, 255), -1)

            # end = time.time()-start
            pre_time = now_time
            # display output
            frame = cv2.putText(frame, "FPS: %3.1f" % (fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = cv2.imshow("Real-time face detection", frame)
            # out.write(video)
            #
            # press "Q" to stop
            if cv2.waitKey(1) == 27:
                goto_origin(pp,ps,tp,ts)
                break


    # release resources
    # out.release()
    vcap.release()
    cv2.destroyAllWindows()
