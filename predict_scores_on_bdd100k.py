import csv
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.backends import cudnn

from builders.dataset_builder import build_dataset_test
from builders.model_builder import build_model
from utils.utils import str_to_bool
from utils.metric.iou import calculate_iou


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="bdd100k", help="dataset: bdd100k")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--save_file_name', type=str, default='iou_results.csv',
                        help='file name of the results')
    parser.add_argument('--cuda', type=str_to_bool, default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args


def calculate_iou_for_all(args, test_loader, model):
    model.eval()
    total_batches = len(test_loader)
    iou_results = []

    for i, (input, label, size, name) in enumerate(test_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = input.cuda()
        else:
            input_var = input

        start_time = time.time()
        with torch.no_grad():
            output_tensor = model(input_var)
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))

        iou, per_class_iu = calculate_iou(torch.argmax(output_tensor, dim=1).detach(), label, args.classes)
        mean_iou = np.mean(iou)

        iou_results.append({'image_path': name[0], 'iou_score': mean_iou})

        # if i == 5:
        #    break
    return iou_results


def check_cuda_available(args):
    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")


def load_model(args):
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    """
    device = torch.device('cpu')
    for i in range(1, 9):
        state_dict = torch.load(f'checkpoint/19895369/ENetbs8gpu1_train/model_{i}.pth', map_location=device)
        model.load_state_dict(state_dict['model'])

        start_time = time.time()
        result = model(torch.ones([1, 3, 720, 1280]))
        time_taken = time.time() - start_time
        print('time: %.2f, mean: %.4f' % (time_taken, result.mean()))

        del state_dict
        del result
    """

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            if not args.cuda:
                device = torch.device('cpu')
            else:
                device = torch.device('cuda')
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    return model


def write_results_to_csv(results, csv_filepath):
    dir_path = os.path.dirname(csv_filepath)
    os.makedirs(dir_path, exist_ok=True)

    with open(csv_filepath, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(['Image Path', 'IoU Score'])

        for result in results:
            csv_writer.writerow([result['image_path'], result['iou_score']])


def predict_scores(args):
    check_cuda_available(args)

    model = load_model(args)
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    print("=====> beginning IoU score file generation")
    print("validation set length: ", len(testLoader))

    iou_results = calculate_iou_for_all(args, testLoader, model)

    csv_filename = os.path.join(args.save_seg_dir, args.save_file_name)
    write_results_to_csv(iou_results, csv_filename)
    print("IoU results saved to:", csv_filename)


if __name__ == '__main__':
    args = parse_args()
    args.classes = 3  # Number of classes in bdd100k dataset

    assert (args.dataset == 'bdd100k')

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)
    predict_scores(args)
