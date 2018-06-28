import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def show_image(image, figsize, title=""):
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title(title)
    ax1.imshow(image)


def add_bounding_boxes(image, detections):
    bounding_boxes_image = np.copy(image)
    for detection in detections:
        for (x,y,w,h) in detection:
            # Add red bounding box for each detection
            cv2.rectangle(bounding_boxes_image, (x,y), (x+w,y+h), (255,0,0), 3)
    return bounding_boxes_image


def show_bounding_boxes(image, detections, figsize, title=""):
    bounding_boxes_image = add_bounding_boxes(image, detections)
    show_image(bounding_boxes_image, figsize=figsize, title=title)
    

def add_blur_to_detections(image, detections):
    blur_image = np.copy(image)
    kernel = np.ones((60, 60),np.float32) / 3200
    for detection in detections:
        for (x,y,w,h) in detection:
            blur_image[y:y+h, x:x+w, :] = cv2.filter2D(blur_image[y:y+h, x:x+w, :], -1, kernel)
    return blur_image


def show_blur_image(image, detections, figsize, title=""):
    blur_image = add_blur_to_detections(image, detections)
    show_image(blur_image, figsize=figsize, title=title)


def _check_cam_stream_shutdown_request():
    key = cv2.waitKey(20)
    if key > 0: # Exit by pressing any key
        # Destroy windows 
        cv2.destroyAllWindows()
            
        # Make sure window closes on OSx
        for i in range (1,5):
            cv2.waitKey(1)
        return True
    return False

def _get_first_frame(vc):
    if vc.isOpened(): 
        rval, frame = vc.read()
        return rval, frame
    else:
        rval = False
        frame = None
        return rval, frame


def cam_stream_manipulation(face_detector, scale_factor, min_neighbors, manipulation_func):
    # Create video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # get first frame
    rval, frame = _get_first_frame(vc)

    # Keep the video stream open
    while rval:
        
        faces = face_detector.detect_faces(frame, scale_factor=scale_factor, min_neighbors=min_neighbors)
        augmented_frame = manipulation_func(frame, (faces,))
        # Plot the image from camera with all the face and eye detections marked
        cv2.imshow("face detection activated", augmented_frame)
        
        # shut down
        if _check_cam_stream_shutdown_request():
            break

        # control framerate for computation
        time.sleep(0.05) 
        # update frame
        rval, frame = vc.read()  