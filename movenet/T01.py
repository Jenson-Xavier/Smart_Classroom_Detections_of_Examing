import tensorflow as tf
import tensorflow_hub as hub

module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

image_path = '../data/images/635a27dae1036588.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image)

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
keypoints_with_scores = movenet(input_image)

print(keypoints_with_scores)
