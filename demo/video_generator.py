# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_folder', help='Image file')
    parser.add_argument('start_index', help='start_index of video')
    parser.add_argument('num_frames', help='number of video frames')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    image_paths = os.listdir(args.img_folder)
    for i in range(int(args.start_index), int(args.start_index)+int(args.num_frames),1):
        img_path = os.path.join(args.img_folder, image_paths[i])
        result = inference_detector(model, img_path)
        out_path = args.out_file + f"_{i}.png"
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=out_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
