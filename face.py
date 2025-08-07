import cv2
import os

# Create folder if it doesn't exist
if not os.path.exists('stored-faces'):
    os.makedirs('stored-faces')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

i = 0  # Global face image counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over faces
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face
        face_img = frame[y:y+h, x:x+w]

        # Save cropped face
        target_file_name = f'stored-faces/face_{i}.jpg'
        cv2.imwrite(target_file_name, face_img)
        print(f"Saved {target_file_name}")
        i += 1

    # Show video feed
    cv2.imshow("Face Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
