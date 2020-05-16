import time
from urllib.request import urlopen

from functools import wraps

import pyfakewebcam
import cv2
import numpy as np

time.sleep(2.0)

camera = pyfakewebcam.FakeWebcam('/dev/video0', 640, 480)


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def resize_image(img, size=(640, 480)):
    h, w, c = img.shape[:3]

    # if size[0] - w > size[1] - h:
    ratio = size[0] / size[1]
    img = cv2.copyMakeBorder(img, 0, 0, int(((w * ratio) - w) / 2), int(((w * ratio) - w) / 2), cv2.BORDER_CONSTANT)

    # return img

    h, w = img.shape[:2]
    # if h == w:
    return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time.time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time.time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it


data = b''
@measure
def read_img(resp):
    global data
    while True:
        data += resp.read(30000)
        find = data.find(b'--frame', -29999)
        if find != -1:
            image_data = data[37:find - 2]
            # image = np.asarray(bytearray(image_data), dtype="uint8")
            data = data[find:]
            # yield cv2.imdecode(image, -1)
            bytes = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), -1)[..., ::-1]
            return bytes


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    resp = urlopen(url)

    while True:
        yield read_img(resp)

    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, readFlag)

    # return the image
    # return image


if __name__ == "__main__":

    picture_gen = url_to_image("http://192.168.1.10:8080/video_feed")
    while True:
        main_picture = next(picture_gen)
        # resized = resize_image(convert(main_picture, 0, 255, np.uint8), (640, 480))

        camera.schedule_frame(main_picture)

        # cv2.imshow('resized', main_picture)

cv2.destroyAllWindows()
