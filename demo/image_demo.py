from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="demo.jpg", help='Image file')
    parser.add_argument('--config',
                        # default="/home/jaychen/Desktop/PycharmProjects/2020.12.10/mmdetection_2.7/configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py",
                        default="/home/jaychen/Desktop/PycharmProjects/2020.12.10/mmdetection_2.7/configs/rpn/rpn_r50_fpn_2x_coco.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        # default="/home/jaychen/Desktop/MODELWEIGHTS/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth",
                        default="../checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth",
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # show RPN result
    img_show = mmcv.imread(args.img)
    mmcv.imshow_bboxes(img_show, result, top_k=20)


if __name__ == '__main__':
    main()
