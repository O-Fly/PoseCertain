import numpy as np
from imgaug import augmenters as iaa
import utils
from ursonet.data.formatting import compose_image_meta, mold_image

def load_image_gt(dataset, config, image_id):
    """Load an image + object pose and apply augmentation pipeline (if necessary)

    Returns:
    image: [height, width, n]
    shape: the original shape of the image before resizing and cropping.
    loc: [x,y,z]
    ori: orientation representation
    """
    # Load and resize image
    image = dataset.load_image(image_id)

    if config.REGRESS_LOC:
        loc = dataset.load_location(image_id)
    else:
        loc = dataset.load_location_encoded(image_id)

    if config.REGRESS_KEYPOINTS:
        keypoints = dataset.load_keypoints(image_id)
        k1 = keypoints[0]
        k2 = keypoints[1]

    if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
        if config.ORIENTATION_PARAM == 'quaternion':
            ori = dataset.load_quaternion(image_id)
        elif config.ORIENTATION_PARAM == 'euler_angles':
            ori = dataset.load_euler_angles(image_id)
        elif config.ORIENTATION_PARAM == 'angle_axis':
            ori = dataset.load_angle_axis(image_id)
    else:
        ori = dataset.load_orientation_encoded(image_id)

    if config.SIM2REAL_AUG:
        image_gray = 0.2126*image[:,:,0]+0.7152*image[:,:,1]+0.0722*image[:,:,2]
        image[:, :, 0] = image_gray
        image[:, :, 1] = image_gray
        image[:, :, 2] = image_gray
        if np.random.rand(1) > 0.5:
            # Image Augmentation Pipeline
            aug_pipeline = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.01 * 255),
                iaa.GaussianBlur(sigma=(0.0,1.5)),
                iaa.Add((-20, 20)),
                iaa.Multiply((0.5,2.0)),
                iaa.CoarseDropout([0.0, 0.03], size_percent=(0.02,0.1))
            ], random_order=True)

            det = aug_pipeline.to_deterministic()
            image = det.augment_image(image)


    if config.ROT_AUG or config.ROT_IMAGE_AUG:
        assert config.REGRESS_LOC
        assert config.ORIENTATION_PARAM == 'quaternion'

        # TODO: The 2 rotation augmentation operations are so far applied with mutual exclusion. Arbitrary may lead to more variation.

        dice = np.random.rand(1)

        # Camera orientation perturbation half the time
        if config.ROT_AUG and dice > 0.5:
            if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
                image, loc, ori = utils.rotate_cam(image, loc, ori, dataset.camera.K, 20)
                k1, k2 = utils.encode_as_keypoints(ori, loc)
            else:
                ori = dataset.load_quaternion(image_id)
                image, loc, ori = utils.rotate_cam(image, loc, ori, dataset.camera.K, 20)

                # Update encoded orientation
                ori = utils.encode_ori_fast(ori, config.BETA, dataset.ori_histogram_map, dataset.ori_output_mask)

        elif config.ROT_IMAGE_AUG and dice <= 0.5:
            if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
                image, loc, ori = utils.rotate_image(image, loc, ori, dataset.camera.K)
                k1, k2 = utils.encode_as_keypoints(ori, loc)
            else:
                ori = dataset.load_quaternion(image_id)
                image, loc, ori = utils.rotate_image(image, loc, ori, dataset.camera.K)

                # Update encoded orientation
                ori = utils.encode_ori_fast(ori, config.BETA, dataset.ori_histogram_map, dataset.ori_output_mask)

    original_shape = image.shape

    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale)

    if config.REGRESS_KEYPOINTS:
        return image, image_meta, loc, k1.T, k2.T
    else:
        return image, image_meta, loc, ori

def data_generator(dataset, config, shuffle=True, batch_size=1):
    """A generator that returns images and corresponding groundtruth.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - gt_locs: [batch, N]
    - gt_oris: [batch, N]
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    tensor_dtype = np.float32
    # For modern GPUs
    if config.F16:
        tensor_dtype = np.float16

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT for image.
            image_id = image_ids[image_index]
            if config.REGRESS_KEYPOINTS:
                image, image_meta, gt_loc, gt_k1, gt_k2 = load_image_gt(dataset, config, image_id)
            else:
                image, image_meta, gt_loc, gt_ori = load_image_gt(dataset, config, image_id)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=tensor_dtype)

                if config.REGRESS_LOC:
                    batch_gt_locs = np.zeros((batch_size, 3), dtype=tensor_dtype)
                else:
                    batch_gt_locs = np.zeros((batch_size, config.LOC_BINS_PER_DIM ** 3), dtype=tensor_dtype)

                if config.REGRESS_KEYPOINTS:
                    batch_gt_k1 = np.zeros((batch_size, 3), dtype=tensor_dtype)
                    batch_gt_k2 = np.zeros((batch_size, 3), dtype=tensor_dtype)
                else:
                    if config.REGRESS_ORI:
                        if config.ORIENTATION_PARAM == 'quaternion':
                            batch_gt_oris = np.zeros((batch_size, 4), dtype=tensor_dtype)
                        else:
                            batch_gt_oris = np.zeros((batch_size, 3), dtype=tensor_dtype)
                    else:
                        batch_gt_oris = np.zeros((batch_size, config.ORI_BINS_PER_DIM ** 3), dtype=tensor_dtype)

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_images[b] = mold_image(image.astype(tensor_dtype), config)
            batch_gt_locs[b] = gt_loc

            if config.REGRESS_KEYPOINTS:
                batch_gt_k1[b] = gt_k1
                batch_gt_k2[b] = gt_k2
            else:
                batch_gt_oris[b] = gt_ori

            b += 1

            # Batch full?
            if b >= batch_size:
                if config.REGRESS_KEYPOINTS:
                    inputs = [batch_images, batch_image_meta, batch_gt_locs, batch_gt_k1, batch_gt_k2]
                else:
                    inputs = [batch_images, batch_image_meta, batch_gt_locs, batch_gt_oris]


                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

