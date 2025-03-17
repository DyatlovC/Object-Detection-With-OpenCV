import cv2
capture = cv2.VideoCapture("") ## Set your video path 

object_dector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=230)

while True:
    ret, frame = capture.read()

    height, width, _ = frame.shape
    print(height, width)

    mask = object_dector.apply(frame)

    countours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 340:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
