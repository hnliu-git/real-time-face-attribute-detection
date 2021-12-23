import cv2
import yaml
import torch
import numpy as np
import face_recognition

from PIL import Image
from dataset import valid_transform
from model import MultiBinMobileNet, MultiTagMobileNet

# Load configurations
cfgs = yaml.load(open("configs/app.yaml"), Loader=yaml.FullLoader)
labels = [label.strip() for label in open(cfgs['label_path'], 'r').readlines()]
reduce_level = int(cfgs['reduce_level'])

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

frame_counter = 0
face_attr_dict = {}

# Bool for processing other frame
process_this_frame = True

if cfgs['tagging']:
    model = MultiTagMobileNet.load_from_checkpoint(cfgs['model_path'], lr=None).cuda().eval()
else:
    model = MultiBinMobileNet.load_from_checkpoint(cfgs['model_path'], lr=None).cuda().eval()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Shape of the frame
    s_h, s_w, _ = frame.shape

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1 / reduce_level, fy=1 / reduce_level)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        frame_counter += 1
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Display the results
        for face_id, (top, right, bottom, left) in enumerate(face_locations):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= reduce_level
            right *= reduce_level
            bottom *= reduce_level
            left *= reduce_level

            h = bottom - top
            w = right - left

            # Add more margin of the box
            top = int(max(top - h / int(cfgs['alpha']), 0))
            left = int(max(left - w / int(cfgs['beta']), 0))
            bottom = int(min(bottom + h / int(cfgs['beta']), s_h - 1))
            right = int(min(right + w / int(cfgs['beta']), s_w - 1))

            rgb_frame_prediction = frame[:, :, ::-1]
            cropped_rgb_frame_prediction = rgb_frame_prediction[top:bottom, left:right]
            frame_tensor = valid_transform(Image.fromarray(cropped_rgb_frame_prediction)).unsqueeze(0).cuda()

            if frame_counter % 100 == 0:
                output = model.forward(frame_tensor)

                if cfgs['tagging']:
                    prediction = [np.squeeze(i) for i in
                                  np.array_split(np.array(output.cpu() > float(cfgs['threshold']), dtype=float), 40, 1)]
                else:
                    prediction = [torch.argmax(i, dim=1).cpu() for i in output]

                temp_attr = []
                for attr_index in range(len(prediction)):
                    attr = int(prediction[attr_index])
                    if attr:
                        temp_attr.append(labels[attr_index].replace("_", " "))
                face_attr_dict[face_id] = temp_attr

            # ---------------draw box and text on frame--------------#
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face

            font = cv2.FONT_HERSHEY_DUPLEX
            try:
                if not "Female" in face_attr_dict[face_id]:
                    if not "Male" in face_attr_dict[face_id]:
                        face_attr_dict[face_id].append("Female")
                # for i in range(0, len(face_attr_dict[face_id]), 3):
                #     face_attr_str = ", ".join(face_attr_dict[face_id][i:i + 3])
                #     cv2.putText(frame, face_attr_str, (left - 30, bottom + i * 5 + 10), font, 0.6, (255, 255, 255), 1)
                for i in range(0, len(face_attr_dict[face_id])):
                    face_attr_str = ", ".join(face_attr_dict[face_id][i:i + 1])
                    cv2.putText(frame, face_attr_str, (left -175, top + i * 16), font, 0.6, (255, 255, 255), 1)
            except KeyError:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "New Face Detected: ID " + str(face_id), (left, bottom - 15), font, 0.6,
                            (255, 255, 255), 1)

    # process_this_frame = not process_this_frame
    process_this_frame = True

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()