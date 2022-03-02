import argparse
import pkg_resources

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import generate_mask, mask_image

from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


def detect_image(image, detector, color=(0,0,255), savefig='figures/detections.jpg'):
    # change detector here (only display confident bboxes)
    # detector output should be a List[([l, t, r, b], score, predicted_class)]
    detections = detector.detect_get_box_in(image, box_format='ltrb')

    result = image.copy()
    for detection in detections:
        print(detection)
        bb, score, predicted_class = detection
        l,t,r,b = bb
        cv2.rectangle(result, (l,t), (r,b), color, 5)
        cv2.putText(result, predicted_class, (l+5, t+40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    plt.figure(figsize=(10, 10))
    plt.imshow(result[:, :, ::-1])
    plt.axis('off')
    plt.savefig(savefig)

def generate_masks_sample(image, detector, rng, grid_size=(16, 16), prob_thresh=0.5, color=(255,255,0),
                          num_rows=5, num_cols=3, savefig='figures/masks_sample.jpg'):
    total_imgs = num_rows * num_cols
    image_h, image_w = image.shape[:2]

    all_masked = []
    for _ in range(total_imgs):
        mask = generate_mask(
            image_size=(image_w, image_h),
            grid_size=grid_size,
            prob_thresh=prob_thresh,
            rng=rng)
        masked = mask_image(image, mask)
        all_masked.append(masked)

    # change detector here (only display confident bboxes)
    # detector output should be a List[List[([l, t, r, b], score, predicted_class)]]
    all_detections = detector.detect_get_box_in(all_masked, box_format='ltrb')

    images = []
    for idx, image_detections in enumerate(all_detections):
        result = all_masked[idx].copy()
        for detection in image_detections:
            bb, score, predicted_class = detection
            l,t,r,b = bb
            cv2.rectangle(result, (l,t), (r,b), color, 5)
            cv2.putText(result, predicted_class, (l+5, t+40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
        images.append(result)

    fig = plt.figure(figsize=(15, 15))
    axes = fig.subplots(num_rows,num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(images[i * num_cols + j][:, :, ::-1])
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.savefig(savefig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Input image path', type=str, required=True)
    parser.add_argument('--grid_size', help='Grid size for mask generation. Default: (16, 16)', nargs=2, type=int, default=[16, 16])
    parser.add_argument('--prob_thresh', help='Probability for mask generation. Default: 0.5', type=float, default=0.5)
    parser.add_argument('--num_rows', help='Number of rows for sample masks. Default: 5', type=int, default=5)
    parser.add_argument('--num_cols', help='Number of columns for sample masks. Default: 5', type=int, default=5)
    args = parser.parse_args()

    #Intialise a random number generator
    rng = np.random.default_rng()

    # initialize detector
    detector = ScaledYOLOV4(
        thresh=0.4,
        nms_thresh=0.5,
        bgr=True,
        gpu_device=0,
        model_image_size=608,
        max_batch_size=64,
        half=True,
        same_size=True,
        weights=pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4l-mish_-state.pt'),
        cfg=pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-csp.yaml'))

    image = cv2.imread(args.img_path)
    detect_image(image, detector)
    generate_masks_sample(
        image,
        detector,
        rng,
        grid_size=tuple(args.grid_size),
        prob_thresh=args.prob_thresh,
        num_rows=args.num_rows,
        num_cols=args.num_cols)


if __name__ == '__main__':
    main()
