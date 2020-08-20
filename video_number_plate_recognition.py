import cv2
import time
import numpy as np

start_time = time.time()

# setting initial frame
a = 1


def frame_processing():
    global a
    video_frames = cv2.VideoCapture(0)
    while True:
        # Frame increment
        a += 1
        # Boolean, Frame
        para, frame = video_frames.read()
        # Printing numpy, matrix
        # print(frame)
        # Initializing Face Cascade
        # Download the .xml file and change the path according to your PC
        number_plate_cascade = cv2.CascadeClassifier('self_made.xml')

        # Frame conversion RGB to GRAY scale
        height = frame.shape[0]
        scale = frame.shape[1] / float(height)
        image = cv2.resize(frame, (int(scale * 475), 480))
        cv2.fastNlMeansDenoising(image, None, 20, 10, 7)
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection co-ordinates
        number_plate_detect = number_plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

        extract_number_plate(number_plate_detect, image, frame)

        # 1 ms delay
        key = cv2.waitKey(1)
        # Press e to exit
        if key == ord('e'):
            break


def extract_number_plate(number_plate_detect, image, frame):
    global a
    # Plotting face using rectangle
    for x in number_plate_detect:
        frame = cv2.rectangle(image, (x[0], x[1]), (x[0] + x[2], x[1] + x[3]), (205, 25, 17), 2)
        # Cropping the face frame and saving it in the CWD
        cropped_frame = frame[x[1]:x[1] + x[3], (x[0] - 15):(x[0] + x[2] + 15)]

        name = 'cropped_img' + str(a) + '.png'
        try:
            cv2.imwrite(name, cropped_frame)
        except cv2.error:
            pass
    cv2.imshow('Frame', frame)


frame_processing()
# Frame analysis / Performance Analysis
print('Total Frames =', a)
end_time = time.time()
print('Total_time = ', end_time-start_time)
print('Frames per second', a/(end_time-start_time))
