import argparse
import pkg_resources

import cv2
import matplotlib.pyplot as plt
import numpy as np
from codetiming import Timer
from tqdm import tqdm

from utils.utils import generate_mask, mask_image, similarity_metric

from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


@Timer(name="generate_saliency_map", text=f"Total time: {{:.4f}} seconds")
def generate_saliency_map(image, detector, target_cls_idx, target_box, num_classes, rng,
                          prob_thresh=0.5, grid_size=(16, 16), n_masks=5000, batch=True, max_batch=1000):
    t1 = Timer("generate_masks", logger=None)
    t2 = Timer("obj_detection", logger=None)
    t3 = Timer("postprocess", logger=None)

    image_h, image_w = image.shape[:2]

    target_one_hot = np.zeros(num_classes)
    target_one_hot[target_cls_idx] = 1

    mask_score = np.empty((image_h, image_w), dtype=np.float32)
    
    if batch:
        for i in range(0, n_masks, max_batch):
            num_masks_batch = min(i+max_batch, n_masks) - i

            all_masks = np.empty((num_masks_batch, image_h, image_w), dtype=np.float32)
            all_masked = []
            with t1:
                for idx in tqdm(range(num_masks_batch),
                                desc=f"Generating masks for [{i}:{min(i+max_batch, n_masks)}]"):
                    mask = generate_mask(
                        image_size=(image_w, image_h),
                        grid_size=grid_size,
                        prob_thresh=prob_thresh,
                        rng=rng)
                    masked = mask_image(image, mask)
                    all_masks[idx, :, :] = mask
                    all_masked.append(masked)
            
            with t2:
                # change detector here (keep all bboxes regardless of confidence or class)
                # detector output should be a List[List[([l, t, r, b], score, predicted_class)]]
                all_detections = detector.detect_get_box_in(all_masked, box_format='ltrb', raw=True)

            with t3:
                all_scores = np.empty(num_masks_batch, dtype=np.float32)
                for idx, mask_detections in enumerate(all_detections):
                    all_scores[idx] = similarity_metric(target_box, target_one_hot, mask_detections)
                mask_score_batch = np.tensordot(all_masks, all_scores, axes=(0,0))

            mask_score += mask_score_batch

    else:
        for _ in tqdm(range(n_masks)):
            with t1:
                mask = generate_mask(
                    image_size=(image_w, image_h),
                    grid_size=grid_size,
                    prob_thresh=prob_thresh,
                    rng=rng)
                masked = mask_image(image, mask)
            
            with t2:
                # change detector here (keep all bboxes regardless of confidence or class)
                # detector output should be a List[([l, t, r, b], score, predicted_class)]
                detections = detector.detect_get_box_in(masked, box_format='ltrb', raw=True)

            with t3:
                score = similarity_metric(target_box, target_one_hot, detections)
                mask_score += mask * score

    print(f'Time taken for generating masks: {Timer.timers.total("generate_masks"):.4f}')
    print(f'Time taken for object detection: {Timer.timers.total("obj_detection"):.4f}')
    print(f'Time taken for postprocessing detections: {Timer.timers.total("postprocess"):.4f}')

    return mask_score

def plot_saliency_map(image, target_box, saliency_map, savefig='figures/saliency.jpg'):
    image_saliency = image.copy()
    cv2.rectangle(image_saliency, tuple(target_box[:2]), tuple(target_box[2:]), (0, 0, 255), 5)
    plt.figure(figsize=(7, 7))
    plt.imshow(image_saliency[:, :, ::-1])
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(savefig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Input image path', type=str, required=True)
    parser.add_argument('--target_box', help='Target bbox coordinates in ltrb', nargs=4, type=int, required=True)
    parser.add_argument('--target_class', help='Target class name', type=str, required=True)
    parser.add_argument('--grid_size', help='Grid size for mask generation. Default: (16, 16)', nargs=2, type=int, default=[16, 16])
    parser.add_argument('--prob_thresh', help='Probability for mask generation. Default: 0.5', type=float, default=0.5)
    parser.add_argument('--masks', help='Number of masks. Default: 1000', type=int, default=1000)
    args = parser.parse_args()

    #Intialise a random number generator
    rng = np.random.default_rng()

    # initialize detector
    detector = ScaledYOLOV4(
        bgr=True,
        gpu_device=0,
        model_image_size=608,
        max_batch_size=64,
        half=True,
        same_size=True,
        weights=pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4l-mish_-state.pt'),
        cfg=pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-csp.yaml'))

    image = cv2.imread(args.img_path)

    saliency_map = generate_saliency_map(
        image,
        detector,
        target_cls_idx=detector.classname_to_idx(args.target_class),
        target_box=args.target_box,
        num_classes = len(detector.class_names),
        rng=rng,
        grid_size=tuple(args.grid_size),
        prob_thresh=args.prob_thresh,
        n_masks=args.masks,
        batch=True)

    plot_saliency_map(image, args.target_box, saliency_map)


if __name__ == '__main__':
    main()
