import cv2
import numpy as np

ARQUIVO_VIDEO = './videos/video_6.mp4'

def calcula_HSI_pele(media):

    H = ((media - 0.15) / 10 - 0.012, (media - 0.15) / 10 + 0.012)
    S = ((media + 0.1) / 10 - 0.070, (media + 0.1) / 10 + 0.070)
    I = (1.12 * media - 0.060, 1.12 * media + 0.060)
    return H, S, I

def detecta_objetos_movimento(imagem_cinza):
    # Aplicar Gaussian blur para reduzir ruído e suavizar a imagem
    blur = cv2.GaussianBlur(imagem_cinza, (7, 7), 0)
    
    # Aplicar limiarização binária para obter objetos em movimento
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    
    # Realizar dilatação e erosão para unir contornos próximos
    dilated = cv2.dilate(thresh, None, iterations=8)
    eroded = cv2.erode(dilated, None, iterations=6)
    
    # Encontrar contornos de objetos em movimento
    objetos, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

     # Fazer uma pequena dilatação para melhorar a detecção de pele
    mascara_pele = cv2.dilate(mascara_pele, None, iterations=1)

    return mascara_pele

def detecta_cabeca(imagem_cinza, x, y, w, h):
    proporcao_cabeca_corpo = 65 / 1000
    raio_estimado = int(h * proporcao_cabeca_corpo)

    circulos_detectados = cv2.HoughCircles(imagem_cinza, cv2.HOUGH_GRADIENT, dp=1.2, minDist=raio_estimado,
                                           param1=50, param2=30, minRadius=raio_estimado-5, maxRadius=raio_estimado+5)

    if circulos_detectados is not None:
        circulos_detectados = np.round(circulos_detectados[0, :]).astype("int")
        for (cx, cy, r) in circulos_detectados:
            cv2.circle(imagem_cinza, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(imagem_cinza, (cx, cy), 2, (0, 0, 255), 3)

# Open the video file or capture from camera
cap = cv2.VideoCapture(ARQUIVO_VIDEO)

# Read the first two frames
_, frame1 = cap.read()
_, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the result to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    objetos = detecta_objetos_movimento(imagem_cinza= gray)

    media_luminosidade = np.mean(gray)

    area_pele = detecta_areas_pele(imagem_hsv=hsv, m=media_luminosidade)

    # Draw the objetos on the original frame
    for objeto in objetos:
        if cv2.contourArea(objeto) < 1000:  # Ignore small objetos
            continue
        (x, y, w, h) = cv2.boundingRect(objeto)

        # Cálculo de α (razão entre largura e altura)
        alpha = w / float(h)

        #desativei a detecção por proporção do ojeto => funciona para alguns frames
        #pro vídeo 5 funciona melhorsinho
        if True:#0.231 <= alpha <= 0.357:
            area_objeto_movimento = area_pele[y:y+h, x:x+w]

            pixels_pele = cv2.countNonZero(area_objeto_movimento)
            total_pixels = area_objeto_movimento.size
            
            #desativei a detecção de pele => praticamente não funciona 
            if True: #pixels_pele / total_pixels > 0.1:

                # Aplicar a Transformada de Hough na região de pele
                detecta_cabeca(gray[y:y+h, x:x+w], x, y, w, h) #deveria desenhar um círculo das cabeças detectadas
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
