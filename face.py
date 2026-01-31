import cv2
import os
import requests

# Server URL
SERVER_URL = "http://192.168.208.44:5000/upload-face"

# Create folder if it doesn't exist
os.makedirs("stored-faces", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
i = 0  # face counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face
        face_img = frame[y:y+h, x:x+w]

        # OPTIONAL: save locally
        local_path = f"stored-faces/face_{i}.jpg"
        cv2.imwrite(local_path, face_img)

        # Encode image for sending
        _, img_encoded = cv2.imencode(".jpg", face_img)

        # Send to server
        response = requests.post(
            SERVER_URL,
            files={"image": img_encoded.tobytes()}
        )

        print(f"Sent face {i} → Server | Status:", response.status_code)
        i += 1

    # Show video feed
    cv2.imshow("Face Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
