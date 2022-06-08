if __name__ == '__main__':
    import cv2
    import numpy as np


    def detect(gray, frame):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if np.any(faces):
            for (x, y, w, h) in faces:
                smiles = []
                # cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 30)
                return frame, faces, smiles
        else:
            return [0,0,0]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    cam = cv2.VideoCapture(0)
    check, img = cam.read()
    res = np.array(img.shape[:2])
    center = res[::-1] // 2
    angles = np.arange(0, 12, 1) * np.pi / 6
    j = 1 / 20

    while True:
        check, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img, face, smile = detect(gray, img)
        if np.any(face):
            face = face[0]
            center = [face[0] + face[2] // 2, face[1] + face[3] // 2]

        angles += j
        if np.any(smile):
            cv2.circle(img, center, 150, (0, 255, 255), 5)
            for i in range(12):
                r = 250
                angle = angles[i]
                p1 = center + (175 * np.array([np.sin(angle), np.cos(angle)])).astype(int)
                p2 = center + (r * np.array([np.sin(angle), np.cos(angle)])).astype(int)
                cv2.line(img, p1, p2, (0, 255, 255), 5)
                cv2.putText(img, 'Sunshine detected', [10, 40], cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 255), thickness=2)
        else:
            cv2.putText(img, 'Smile', [center[0] - 120, center[1] - 100], cv2.FONT_HERSHEY_SIMPLEX, 3, color=(255, 0, 0),
                        thickness=3)

        key = cv2.waitKey(1)

        cv2.imshow('Where`s the little sun? :)', img)
        if key == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

