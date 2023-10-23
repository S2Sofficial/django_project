import cv2
import os

# Create a directory to store the dataset
dataset_dir = 'eye_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Load the pre-trained eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the webcam (you can also use a video file by providing the file path)
cap = cv2.VideoCapture(0)

image_count = 0  # Counter for saved images

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        # Draw rectangles around detected eyes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image with annotated eye positions
        image_count += 1
        image_filename = os.path.join(dataset_dir, f'eye_{image_count}.jpg')
        cv2.imwrite(image_filename, frame)

    cv2.imshow('Eye Dataset Creation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= 100:
        # Press 'q' to quit or set a limit on the number of images to capture
        break

cap.release()
cv2.destroyAllWindows()
