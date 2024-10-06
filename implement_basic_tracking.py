# Pykinect2 Test Code for Python 3.10 with Face Tracking and Infrared
import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Continuously capture frames until 'q' is pressed
while True:
    # Check if a new color frame is available
    if kinect.has_new_color_frame():
        # Get the new color frame as a 1D array
        frame = kinect.get_last_color_frame()

        # Check if the frame contains meaningful data
        if np.any(frame):
            # Reshape the frame into (height, width, 4) - BGRA format
            color_image = frame.reshape((color_height, color_width, 4)).astype(np.uint8)

            # Convert BGRA to BGR for OpenCV
            image_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces and calculate distance
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get the depth value at the center of the face
                depth_frame = kinect.get_last_depth_frame()
                if depth_frame is not None:
                    depth_image = depth_frame.reshape((depth_height, depth_width))
                    face_center_x = int(x + w / 2)
                    face_center_y = int(y + h / 2)

                    if 0 <= face_center_x < depth_width and 0 <= face_center_y < depth_height:
                        depth_value = depth_image[face_center_y, face_center_x]
                        if depth_value != 0:  # Check if the depth value is valid
                            distance = depth_value / 1000.0  # Convert to meters
                            cv2.putText(image_bgr, f"Distance: {distance:.2f} m", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the color image with face detection
            cv2.imshow('Kinect Color Frame', image_bgr)
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