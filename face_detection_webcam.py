import face_recognition
import cv2

from model import MultiBinMobileNet

# Below includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at a smaller resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
# Extent of scaling up the upper bound
alpha = 2
# Extent of scaling up the other bounds
beta = 8
# Extent of reducing to speed up the process
reduce_level = 2
# Bool for processing other frame
process_this_frame = True

model = MultiBinMobileNet.load_from_checkpoint('fx-epoch=07-val_loss=8.2262897.ckpt').cuda().eval()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Shape of the frame
    s_h, s_w, _ = frame.shape

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/reduce_level, fy=1/reduce_level)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= reduce_level
            right *= reduce_level
            bottom *= reduce_level
            left *= reduce_level

            h = bottom - top
            w = right - left

            # Add more margin of the box
            top = int(max(top - h / alpha, 0))
            left = int(max(left - w / beta, 0))
            bottom = int(min(bottom + h / beta, s_h - 1))
            right = int(min(right + w / beta, s_w - 1))

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()