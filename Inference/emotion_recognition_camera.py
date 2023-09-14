# ----------------------------------CONFIG----------------------------------------------------------------

subjects_are_far = False
padding = [0.5, 0.1, 0.2, 0.2]
model = 2
top_emotions = 3

record_button = 114 #q
exit_button = 101 #e
change_model_button = 99 #c


# ------------------------------------CODE-----------------------------------------------------------------


import cv2
import numpy as np
import mediapipe as mp
import torch
from predict import predict, get_transform, next_model, working_model

next_model(model)
transform = get_transform()
mp_face_detection = mp.solutions.face_detection
recording = False
out = None

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=int(subjects_are_far), min_detection_confidence=0.5
) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            face_coords = []
            faces = []

            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                rect_coords = np.array(
                    [
                        [
                            box.xmin - box.height * padding[2],
                            box.ymin - box.height * padding[0],
                        ],
                        [
                            box.xmin + box.width * (1 + padding[3]),
                            box.ymin + box.height * (1 + padding[1]),
                        ],
                    ]
                )
                rect_coords[:, 1] *= image.shape[0]
                rect_coords[:, 0] *= image.shape[1]
                rect_coords = rect_coords.astype(int)
                face = image[
                    rect_coords[0, 1] : rect_coords[1, 1],
                    rect_coords[0, 0] : rect_coords[1, 0],
                ]

                if not np.any(face):
                    continue

                faces.append(transform(face))
                face_coords.append(rect_coords)

                # cv2.imshow('Crop', cv2.cvtColor(image[rect_coords[1,1]:rect_coords[0,1], rect_coords[0,0]:rect_coords[1,0]], cv2.COLOR_BGR2GRAY))

            if faces:
                emotions = predict(torch.stack(faces), top_n=top_emotions)

                for i in range(len(emotions)):
                    image = cv2.rectangle(
                        image, face_coords[i][0], face_coords[i][1], (255, 0, 0), 2
                    )
                    image = cv2.putText(
                        image,
                        emotions[i, 0],
                        face_coords[i][0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    secondary_emotions = "\n".join(emotions[i, 1:])

                    c = face_coords[i][[1, 0], [0, 1]]
                    dy = 50
                    for i, line in enumerate(secondary_emotions.split("\n")):
                        c[1] += i * dy
                        cv2.putText(
                            image,
                            line,
                            c,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (147, 1, 113),
                            2,
                            cv2.LINE_AA,
                        )

        info_text = f"Dataset of Model : {working_model()}\n{chr(change_model_button)}: Change model\n{chr(exit_button)}: Exit\n{chr(record_button)}: Record"

        y0, dy = 20, 25
        for i, line in enumerate(info_text.split("\n")):
            y = y0 + i * dy
            cv2.putText(
                image,
                line,
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Emotion Recognition", image)
        key_pressed = cv2.waitKey(5) & 0xFF
    
        if key_pressed == exit_button:
            break
        elif key_pressed == change_model_button:
            next_model()
            transform = get_transform()

        elif key_pressed == record_button:
            if recording:
                out.release()
            else:
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                out = cv2.VideoWriter(
                    "recording.avi",
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                    cap.get(cv2.CAP_PROP_FPS),
                    (frame_width, frame_height),
                )
            recording = not recording

        if recording:
            out.write(image)

cap.release()
if out:
    out.release()
