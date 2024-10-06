# Pykinect2 Test Code for Python 3.10 with Face Recognition and Tracking
import cv2
import numpy as np
import time
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import face_recognition

# Initialize Kinect runtime for color, depth, and infrared frames
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth |
                                         PyKinectV2.FrameSourceTypes_Infrared)

# Get the color frame width and height
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
# Get the depth frame width and height
depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height
# Get the infrared frame width and height
infrared_width, infrared_height = kinect.infrared_frame_desc.Width, kinect.infrared_frame_desc.Height

# Create windows to display the frames
cv2.namedWindow('Kinect Color Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Kinect Depth Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Kinect Infrared Frame', cv2.WINDOW_NORMAL)

# Load a reference image of your face
reference_image_path = "reference_face.png"  # Replace with the path to your reference image
reference_image = face_recognition.load_image_file(reference_image_path)
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

# Continuously capture frames until 'q' is pressed
frame_skip = 2  # Number of frames to skip to reduce processing load
frame_count = 0

while True:
    frame_count += 1

    # Skip frames to reduce the processing load
    if frame_count % frame_skip != 0:
        time.sleep(0.03)  # Add a small delay to reduce CPU usage
        continue

    # Check if a new color frame is available
    if kinect.has_new_color_frame():
        # Get the new color frame as a 1D array
        frame = kinect.get_last_color_frame()

        # Check if the frame contains meaningful data
        if np.any(frame):
            # Reshape the frame into (height, width, 4) - BGRA format
            color_image = frame.reshape((color_height, color_width, 4)).astype(np.uint8)

            # Convert BGRA to RGB for face recognition
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(image_rgb)
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face to the reference face
                matches = face_recognition.compare_faces([reference_face_encoding], face_encoding)

                if True in matches:
                    # Draw a rectangle around the recognized face
                    cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Get the depth value at the center of the face
                    depth_frame = kinect.get_last_depth_frame()
                    if depth_frame is not None:
                        depth_image = depth_frame.reshape((depth_height, depth_width))
                        face_center_x = int((left + right) / 2 * depth_width / color_width)
                        face_center_y = int((top + bottom) / 2 * depth_height / color_height)

                        if 0 <= face_center_x < depth_width and 0 <= face_center_y < depth_height:
                            depth_value = depth_image[face_center_y, face_center_x]
                            if depth_value != 0:  # Check if the depth value is valid
                                distance = depth_value / 1000.0  # Convert to meters
                                cv2.putText(color_image, f"Distance: {distance:.2f} m", (left, top - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the color image with face recognition
            cv2.imshow('Kinect Color Frame', color_image)
        else:
            print("Color frame contains no valid data (all pixels are zero).")

    # Check if a new depth frame is available
    if kinect.has_new_depth_frame():
        # Get the new depth frame as a 1D array
        frame = kinect.get_last_depth_frame()

        # Check if the frame contains meaningful data
        if np.any(frame):
            # Reshape the frame into (height, width) - depth data is 16-bit
            depth_image = frame.reshape((depth_height, depth_width))

            # Normalize the depth image for visibility
            depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Display the depth image using OpenCV
            cv2.imshow('Kinect Depth Frame', depth_image)
        else:
            print("Depth frame contains no valid data (all pixels are zero).")

    # Check if a new infrared frame is available
    if kinect.has_new_infrared_frame():
        # Get the new infrared frame as a 1D array
        frame = kinect.get_last_infrared_frame()

        # Check if the frame contains meaningful data
        if np.any(frame):
            # Reshape the frame into (height, width) - infrared data is 16-bit
            infrared_image = frame.reshape((infrared_height, infrared_width))

            # Normalize the infrared image for visibility
            infrared_image = cv2.normalize(infrared_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Display the infrared image using OpenCV
            cv2.imshow('Kinect Infrared Frame', infrared_image)
        else:
            print("Infrared frame contains no valid data (all pixels are zero).")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close the windows
cv2.destroyAllWindows()
kinect.close()