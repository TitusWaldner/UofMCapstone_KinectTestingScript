# Pykinect2 Test Code for Python 3.10 with Face Recognition and Body Tracking using MediaPipe
import cv2
import numpy as np
import time
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import face_recognition
import mediapipe as mp
from pykalman import KalmanFilter

# Initialize Kinect runtime for color frames
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

# Get the color frame width and height
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height

# Create windows to display the frames
cv2.namedWindow('Kinect Color Frame', cv2.WINDOW_NORMAL)

# Load a reference image of your face
reference_image_path = "reference_face.png"  # Replace with the path to your reference image
reference_image = face_recognition.load_image_file(reference_image_path)
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize MediaPipe Pose for body tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize Kalman Filter for body tracking
kf = KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=2)
current_position = None

# Continuously capture frames until 'q' is pressed
frame_skip = 2  # Number of frames to skip to reduce processing load
frame_count = 0

while True:
    frame_count += 1

    # Skip frames to reduce the processing load
    if frame_count % frame_skip != 0:
        time.sleep(0.03)  # Add a small delay to reduce CPU usage
        continue

    face_detected = False

    # Check if a new color frame is available
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        # Get the new color frame as a 1D array
        frame = kinect.get_last_color_frame()

        # Check if the frame contains meaningful data
        if np.any(frame):
            # Reshape the frame into (height, width, 4) - BGRA format
            color_image = frame.reshape((color_height, color_width, 4)).astype(np.uint8)

            # Convert BGRA to BGR for MediaPipe and RGB for face recognition
            image_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(image_rgb)
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face to the reference face
                matches = face_recognition.compare_faces([reference_face_encoding], face_encoding)

                if True in matches:
                    face_detected = True

                    # Update the current position with the face position
                    current_position = np.array([int((left + right) / 2), int((top + bottom) / 2)])

                    # Draw a rectangle around the recognized face
                    cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Get the depth value at the center of the face
                    depth_frame = kinect.get_last_depth_frame()
                    depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
                    face_center_x = int((left + right) / 2 * kinect.depth_frame_desc.Width / color_width)
                    face_center_y = int((top + bottom) / 2 * kinect.depth_frame_desc.Height / color_height)
                    if 0 <= face_center_x < kinect.depth_frame_desc.Width and 0 <= face_center_y < kinect.depth_frame_desc.Height:
                        depth_value = depth_image[face_center_y, face_center_x]
                        if depth_value != 0:
                            distance = depth_value / 1000.0  # Convert to meters
                            cv2.putText(color_image, f"Distance: {distance:.2f} m", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If face is not detected, use MediaPipe Pose to track the body
            if not face_detected and current_position is not None:
                results = pose.process(image_bgr)
                if results.pose_landmarks:
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Get the coordinates of the body center (using the nose or torso landmark as a reference)
                    nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    body_center_x = int(nose_landmark.x * color_width)
                    body_center_y = int(nose_landmark.y * color_height)

                    # Update Kalman filter with new observation
                    current_position, _ = kf.filter_update(filtered_state_mean=current_position, filtered_state_covariance=np.eye(2), observation=[body_center_x, body_center_y])

                    # Draw a circle at the predicted body center
                    cv2.circle(color_image, (body_center_x, body_center_y), 10, (255, 0, 0), -1)

                    # Get the depth value at the body center
                    depth_frame = kinect.get_last_depth_frame()
                    depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
                    if 0 <= body_center_x < kinect.depth_frame_desc.Width and 0 <= body_center_y < kinect.depth_frame_desc.Height:
                        depth_value = depth_image[body_center_y, body_center_x]
                        if depth_value != 0:
                            distance = depth_value / 1000.0  # Convert to meters
                            cv2.putText(color_image, f"Distance: {distance:.2f} m", (body_center_x, body_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the color image with face recognition and body tracking
            cv2.imshow('Kinect Color Frame', color_image)
        else:
            print("Color frame contains no valid data (all pixels are zero).")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close the windows
cv2.destroyAllWindows()
kinect.close()