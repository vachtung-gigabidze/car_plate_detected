import cv2 
import pytesseract
import re 

print("start")

# Обрезать изображение
def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    carplate_img = image

    for x, y, w, h in carplate_rects:
        # carplate_img = image[y+15:y+h-10, x+15:x+w-20]
        carplate_img = image[y:y+h, x:x+w]

    return carplate_img

def carplate_rectangle(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    carplate_img = image

    for x, y, w, h in carplate_rects:
        font = cv2.FONT_HERSHEY_SIMPLEX
        carplate_img = cv2.putText(carplate_img,'HOMEP',(x-50,y-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        carplate_img = cv2.rectangle(carplate_img,(x,y),(x+w,y+h),(0,255,0),1)
        # carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)    
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized_image

def catchText(carplate_img_rgb, carplate_haar_cascade):
    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)    

    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)    

    plate = pytesseract.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTXYZ0123456789')
    cv2.imshow('gray', carplate_extract_img_gray) 
   
    if (bool(re.search(r'^[A-Z]{0,1}\s{0,}\d{3,4}\s{0,}[A-Z]{0,2}\s{0,}\d{2,4}$', plate))):
        print('Номер авто: ', plate
        )

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
carplate_haar_cascade = cv2.CascadeClassifier("C:\\Projects\\python\\CAR_PLATE_DETECTED\\haar_cascades\\haarcascade_licence_plate_rus_16stages.xml")  
while(True): 
      
    
    ret, frame = vid.read() 
    carplate_extract_img = carplate_rectangle(frame, carplate_haar_cascade)
    
    catchText(frame, carplate_haar_cascade)
    cv2.imshow('carplate_extract_img', carplate_extract_img) 
      
   
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

vid.release() 

cv2.destroyAllWindows() 