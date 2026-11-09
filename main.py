import os
import cv2
import face_recognition
from deepface import DeepFace

FACES_DIR = "known_faces"
UNKNOWN_LABEL = "Unknown Person"
TOLERANCE = 0.5
EMOTION_INTERVAL = 10


class FaceRegistry:
    def __init__(self, folder):
        self.folder = folder
        self.names = []
        self.encodings = []
        self.load()

    def load(self):
        self.names.clear()
        self.encodings.clear()
        if not os.path.isdir(self.folder):
            return
        for person in os.listdir(self.folder):
            person_path = os.path.join(self.folder, person)
            if not os.path.isdir(person_path):
                continue
            for file_name in os.listdir(person_path):
                image = face_recognition.load_image_file(os.path.join(person_path, file_name))
                found = face_recognition.face_encodings(image)
                if found:
                    self.names.append(person)
                    self.encodings.append(found[0])

    def identify(self, encoding):
        if not self.encodings:
            return UNKNOWN_LABEL
        distances = face_recognition.face_distance(self.encodings, encoding)
        best = distances.argmin()
        if distances[best] <= TOLERANCE:
            return self.names[best]
        return UNKNOWN_LABEL


class EmotionDetector:
    def detect(self, face_image):
        if face_image.size == 0:
            return "unknown"
        try:
            result = DeepFace.analyze(
                face_image,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
            )
            return result[0]["dominant_emotion"]
        except Exception:
            return "unknown"


def register_user(name):
    folder = os.path.join(FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)
    count = len(os.listdir(folder))
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        display = frame.copy()
        text = f"{name} | saved: {count} | S = save, Q = finish"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Register", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), frame)
            count += 1
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def run_recognition():
    registry = FaceRegistry(FACES_DIR)
    emotion_detector = EmotionDetector()
    camera = cv2.VideoCapture(0)
    frame_count = 0
    emotions = []

    while True:
        success, frame = camera.read()
        if not success:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        boxes = [(top * 4, right * 4, bottom * 4, left * 4) for top, right, bottom, left in locations]

        if frame_count % EMOTION_INTERVAL == 0 or len(emotions) != len(boxes):
            emotions = []
            for top, right, bottom, left in boxes:
                emotions.append(emotion_detector.detect(frame[top:bottom, left:right]))

        for (top, right, bottom, left), encoding, emotion in zip(boxes, encodings, emotions):
            name = registry.identify(encoding)
            color = (0, 0, 255) if name == UNKNOWN_LABEL else (0, 255, 0)
            label = f"{name} - {emotion}"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition and Emotion Detection", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("1. Register new user")
        print("2. Start recognition")
        print("3. Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            name = input("Enter name: ").strip()
            if name:
                register_user(name)
        elif choice == "2":
            run_recognition()
        elif choice == "3":
            break


if __name__ == "__main__":
    main()