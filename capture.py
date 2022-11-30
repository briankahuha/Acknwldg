import cv2 as cv

cam = cv.VideoCapture(0)

cv.namedWindow("Capture")

img_counter = 0

schID = str(651436)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv.imshow("test", frame)

    k = cv.waitKey(1)

    if k%256 == 27:
        break
    elif k%256 == 32:
        while img_counter <= 40:
            img_name = "{}.png".format(img_counter)
            cv.imwrite("SampleImages\ "+schID+"-"+img_name, frame)
            img_counter += 1

cam.release()