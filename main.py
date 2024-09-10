import cv2
import numpy as np

# Open the video file or capture from camera
cap = cv2.VideoCapture('./videos/video_1.mp4')

# Read the first two frames
_, frame1 = cap.read()
_, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the result to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and smoothen the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to obtain the moving objects
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Perform dilation to fill gaps in the detected objects
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours of the moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small contours
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (w + x, h + y), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Pedestrian Detection", frame1)

    # Update the frames for the next iteration
    frame1 = frame2
    _, frame2 = cap.read()

    # Break the loop with the 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()