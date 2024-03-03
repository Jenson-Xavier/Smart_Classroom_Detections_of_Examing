from movenet import MoveNet, draw_prediction_on_image
import cv2
import tensorflow as tf
import numpy as np

mnet = MoveNet()
cap = cap = cv2.VideoCapture('../data/videos/TestVideo03.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints_with_scores = mnet.run(frame)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(frame, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    cv2.imshow('video', output_overlay)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()