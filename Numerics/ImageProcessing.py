import numpy as np

def z_standardise(pixels):
    mean = np.mean(pixels)
    std_dev = np.std(pixels)

    return (pixels - mean) / std_dev


def normalize_to_image_space(pixels):
    pixels_min = np.min(pixels)
    pixels_max = np.max(pixels)

    # https://en.wikipedia.org/wiki/Normalization_(image_processing)
    mapped_float = (pixels - pixels_min) * (255.0 / (pixels_max - pixels_min))
    return mapped_float.astype(np.uint8)

