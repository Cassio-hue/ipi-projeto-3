import cv2
import numpy as np

def calcula_HSI_pele(media):

    H = ((media - 0.15) / 10 - 0.012, (media - 0.15) / 10 + 0.012)
    S = ((media + 0.1) / 10 - 0.070, (media + 0.1) / 10 + 0.070)
    I = (1.12 * media - 0.060, 1.12 * media + 0.060)
    return H, S, I

def detecta_objetos_movimento(imagem_cinza):
    # Apply Gaussian blur to reduce noise and smoothen the image
    blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    # Apply binary thresholding to obtain the moving objects
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Perform dilation to fill gaps in the detected objects
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find objetos of the moving objects
    objetos, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return objetos

def detecta_areas_pele(imagem_hsv, m):
    canal_H = imagem_hsv[:, :, 0]
    canal_S = imagem_hsv[:, :, 1]
    canal_I = imagem_hsv[:, :, 2]
    
    intervalo_H_pele, intervalo_S_pele, intervalo_I_pele = calcula_HSI_pele(media=m)

    mascara_H = cv2.inRange(canal_H, int(intervalo_H_pele[0] * 179), int(intervalo_H_pele[1] * 179))
    mascara_S = cv2.inRange(canal_S, int(intervalo_S_pele[0] * 255), int(intervalo_S_pele[1] * 255))
    mascara_I = cv2.inRange(canal_I, int(intervalo_I_pele[0] * 255), int(intervalo_I_pele[1] * 255))

    mascara_pele =  cv2.bitwise_and(mascara_H, mascara_S)
    mascara_pele = cv2.bitwise_and(mascara_pele, mascara_I)

    return mascara_pele

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
    hsv = cv2. cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    objetos = detecta_objetos_movimento(imagem_cinza= gray)

    media_luminosidade = np.mean(gray)

    area_pele = detecta_areas_pele(imagem_hsv=hsv, m=media_luminosidade)

    # Draw the objetos on the original frame
    for objeto in objetos:
        if cv2.contourArea(objeto) < 500:  # Ignore small objetos
            continue
        (x, y, w, h) = cv2.boundingRect(objeto)

        area_objeto_movimento = area_pele[y:y+h, x:x+w]

        pixels_pele = cv2.countNonZero(area_objeto_movimento)
        total_pixels = area_objeto_movimento.size
        
        if pixels_pele / total_pixels > 0.01:
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
