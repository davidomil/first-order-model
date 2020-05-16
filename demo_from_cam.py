import time
import pyfakewebcam
import matplotlib
import cv2

matplotlib.use('Agg')
import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

vs = cv2.VideoCapture(2)
print(cv2.CAP_PROP_FPS)

time.sleep(2.0)

camera = pyfakewebcam.FakeWebcam('/dev/video0', 640, 480)


def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def imgCrop(image, cropBox, boxScale=1.4):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta = int(max(cropBox[2] * (boxScale - 1), 0))
    yDelta = int(max(cropBox[3] * (boxScale - 1), 0))

    # # Convert cv box to PIL box [left, upper, right, lower]
    # PIL_box = [cropBox[0] - xDelta, cropBox[1] - yDelta, cropBox[0] + cropBox[2] + xDelta,
    #            cropBox[1] + cropBox[3] + yDelta]

    return image[cropBox[1] - yDelta: cropBox[1] + yDelta + cropBox[3],
           cropBox[0] - xDelta:cropBox[0] + xDelta + cropBox[2]]


def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    # Equalize the histogram
    image = cv2.equalizeHist(image)

    # Detect the faces
    # faceCascade.load('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30))

    # If faces are found
    if len(faces) > 0 and returnImage:
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = image[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

    if returnImage:
        return image
    else:
        return faces


last_faces = []


def get_frame():
    global last_faces
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = vs.read()
        # original_frame = frame.copy()
        croppedImage = []
        if len(last_faces) > 0:
            # for (x, y, w, h) in last_faces:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            croppedImage = imgCrop(frame, last_faces[0])
        else:
            faces = DetectFace(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), faceCascade)
            if len(faces) > 0:
                croppedImage = imgCrop(frame, faces[0])
                last_faces = faces.copy()

        yield resize(croppedImage if len(croppedImage) > 0 else frame, (256, 256))[..., :3]


force_initial_kp = None

frame_generator = get_frame()


def make_animation(source_image, generator, kp_detector, relative=True, adapt_movement_scale=True):
    with torch.no_grad():
        global frame_generator
        global force_initial_kp
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        kp_source = kp_detector(source)
        force_initial_kp = kp_detector(
            torch.tensor(next(frame_generator)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda())

        for frame in frame_generator:
            driving_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=force_initial_kp, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            # kp_driving_initial = kp_driving
            main_picture = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            yield main_picture, np.hstack(
                (main_picture, np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0]))


def resize_image(img, size=(640, 480)):
    h, w, c = img.shape[:3]

    # if size[0] - w > size[1] - h:
    ratio = size[0] / size[1]
    img = cv2.copyMakeBorder(img, 0, 0, int(((w*ratio) - w)/2), int(((w*ratio) - w)/2), cv2.BORDER_CONSTANT)

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


def find_best_frame(source, driving):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)

    source_image = resize(source_image, (256, 256))[..., :3]
    # driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)

    animation_generator = make_animation(source_image, generator, kp_detector, relative=opt.relative,
                                         adapt_movement_scale=opt.adapt_scale)

    while True:

        main_picture, predictions = next(animation_generator)
        resized = resize_image(convert(main_picture, 0, 255, np.uint8), (640, 480))
        camera.schedule_frame(resized)

        cv2.imshow('resized', resized)
        cv2.imshow('frame', predictions)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('s'):
            force_initial_kp = kp_driving_initial = kp_detector(
                torch.tensor(next(frame_generator)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda())
            print("New Main Frame set")

        if k == ord('r'):
            last_faces = []
            print("Face reset")

    # predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)

vs.release()
cv2.destroyAllWindows()
