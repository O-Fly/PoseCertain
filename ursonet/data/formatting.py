import numpy as np

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image coords
        [scale]  # size=1
    )
    return meta


def mold_image(image, config):
    """Subtract the mean pixel and converts it to float.
    """

    tensor_dtype = np.float32
    if config.F16:
        tensor_dtype = np.float16

    if image.shape[-1] == 3:
        return image.astype(tensor_dtype) - config.MEAN_PIXEL
    else:
        return image.astype(tensor_dtype) - np.mean(config.MEAN_PIXEL)


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original.
    TODO: This does not accept grayscale"""

    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

