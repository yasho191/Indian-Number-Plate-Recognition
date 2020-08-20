import cv2
import numpy as np


def image_processing(photo_paths):
    global a
    for j in range(len(photo_paths)):
        path = photo_paths[j][:-1]

        # Download the .xml file and change the path
        number_plate_cascade = cv2.CascadeClassifier('self_made.xml')

        # image refining and resizing for optimal result
        image = cv2.imread(path, 1)
        height = image.shape[0]
        scale = image.shape[1] / float(height)
        image = cv2.resize(image, (int(scale * 713), 720))
        cv2.fastNlMeansDenoising(image, None, 20, 10, 7)
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Finding out the coordinates of the number plate region
        search_number_plate = number_plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

        # Number plate extraction plotting and cropping the detected region
        number_plate_extract(search_number_plate, image)


def number_plate_extract(search_number_plate, image):
    global a
    if type(search_number_plate) != type(()):
        for x in search_number_plate:
            a += 1
            image = cv2.rectangle(image, (x[0], x[1]), (x[0] + x[2], x[1] + x[3]), (205, 25, 17), 3)
            name = str(a) + '.png'
            cropped_img = image[x[1]:x[1] + x[3], (x[0]-15):(x[0] + x[2]+15)]
            cv2.imwrite(name, cropped_img)

        cv2.imshow('final_image', image)

        print('Number Plate has been detected')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('No Number Plate Detected')


a = 0
photo_paths = []
# Create a file with paths on your PC
file = open('paths', 'r')
for i in file.readlines():
    photo_paths.append(i)

# Function call for start of the process
image_processing(photo_paths)
