import cv2
try:
    trained_face_data = cv2.CascadeClassifier(r'C:\Users\ishaa\human-facial-recognition\haarcascade_frontalface_default.xml')

    
    vid = cv2.VideoCapture(0)
    while True:
        ret, img= vid.read()
        grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
        for i in range(len(face_coordinates)):
            
            cv2.rectangle(img,(face_coordinates[i][0], face_coordinates[i][1]),(face_coordinates[i][0] + face_coordinates[i][2], face_coordinates[i][1] +face_coordinates[i][2] ),(0, 225,0),2)
            
        cv2.imshow("ishaan face detector",img  )

        
        
        if cv2.waitKey(1) & 0xFF == ord("q") :
            break
        

    
    
except Exception as e:
    print(e)
print("code completed")
# print(face_coordinates)