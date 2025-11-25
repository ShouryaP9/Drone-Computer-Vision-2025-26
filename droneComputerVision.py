import cv2
import numpy as np

#connects to camera
cam = cv2.VideoCapture(0)
#Defines the range of colors that should be detected in the green-red spectrum (in this case, shades of red)
redThreshold = 163
#Defines the range of colors that should be detected in the blue-yellow spectrum (in this case, not shades of blue)
blueThreshold = 136
#Minimum area of a detected object for it to be circled
minArea = 6000
largestShape = None
#Used for reducing noise and filling holes in object detection
kernel = np.ones((3, 3), np.uint8)

#Runs the code inside continuously
while True:
    #reads the camera input
    _, frame = cam.read()

    #blurs the camera input to reduce noise and glare
    blurredFrame = cv2.GaussianBlur(frame, (7,7), 0)

    #Converts to LAB color space (better for color detection)
    lab = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2LAB)

    #splits the LAB channels and assigns A (red and green detection) and B (blue and yellow detection) to variables
    _, A, B = cv2.split(lab)

    #Creates a mask to show where red is detected
    _, maskA = cv2.threshold(A, redThreshold, 255, cv2.THRESH_BINARY)
    
    #Creates a mask to show where blue isn't detected
    _, maskB = cv2.threshold(B, blueThreshold, 255, cv2.THRESH_BINARY)

    #Combines the two masks to detect red and exclude blue
    mask = cv2.bitwise_and(maskA, maskB)

    #gets the shapes/contours of red detected by the mask, using the basic contour finding mode and method
    shapes, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #reduces noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #fills holes in detected objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if len(shapes) > 0:
        for s in shapes:
            if cv2.contourArea(s) >= minArea:

                #returns a dictionary of moments
                moment = cv2.moments(s)

                if moment["m00"] != 0:
                    #x coordinate of center of detected object
                    x = int(moment["m10"] / moment["m00"])
                    #y coordinate of center of detected object
                    y = int(moment["m01"] / moment["m00"])

                    #circles the detected object with a circle of radius 6, color green, and thickness of 6
                    cv2.circle(frame, (x, y), 100, (0, 255, 0), 7)

    #Shows the normal camera feed and the mask
    cv2.imshow("camera", frame)
    cv2.imshow("red detection", mask)

    # Breaks the while loop if escape is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
