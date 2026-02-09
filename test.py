import os
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.evaluation import PixelF1

from IMDLBenCo.training_scripts.tester import test_one_epoch

from conv_cswin import Mesorch_ConvNeXt_CSWinB



def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo testing launch!', add_help=True)

    parser.add_argument('--model', default=None, type=str, required=True,
                        help='The name of applied model')
    parser.add_argument('--if_predict_label', action='store_true',
                        help='If model uses label for training')

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--if_padding', action='store_true')
    parser.add_argument('--if_resizing', action='store_true')

    parser.add_argument('--edge_mask_width', default=None, type=int)

    parser.add_argument('--test_data_json', default='./test_data.json', type=str,
                        help='a JSON dict: {dataset_name: path} ')

    parser.add_argument('--checkpoint_path', default='./output_dir', type=str)
    parser.add_argument('--test_batch_size', default=2, type=int)
    parser.add_argument('--no_model_eval', action='store_true')

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed parameters (we disable them)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)

    args, remaining_args = parser.parse_known_args()

    # build model arg parser
    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args


def main(args, model_args):
    # ---- disable distributed ----
    args.distributed = False
    args.gpu = 0

    print("===== args =====")
    print(args)
    print("===== model args =====")
    print(model_args)

    device = torch.device(args.device)

    # ---- load test sets JSON ----
    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)

    if not isinstance(test_dataset_json, dict):
        raise TypeError(
            "❌ test_data_json 必须是一个 dict，例如:\n"
            "{\n"
            "  \"CASIA1\": \"/path/to/CASIA1\",\n"
            "  \"Columbia\": \"/path/to/Columbia\"\n"
            "}"
        )

    # ---- model init ----
    model = MODELS.get(args.model)

    # select needed init args
    if isinstance(model, (types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters

    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})

    model = model(**combined_args)
    model.to(device)

    print("Model = ", model)

    # ---- evaluator ----
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
    ]

    # ---- transforms ----
    test_transform = get_albu_transforms('test')

    # ---- post function ----
    post_function_name = f"{args.model}_post_func".lower()
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None

    # ---- iterate datasets ----
    start_time = time.time()

    for dataset_name, dataset_path in test_dataset_json.items():
        print(f"\n==============================")
        print(f"Start testing dataset: {dataset_name}")
        print("==============================")

        # prepare log dir
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)
        os.makedirs(args.full_log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.full_log_dir)

        # ---- load dataset ----
        if os.path.isdir(dataset_path):
            # folder → ManiDataset
            dataset_test = ManiDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        else:
            # JSON → either JsonDataset or BalancedDataset
            try:
                dataset_test = JsonDataset(
                    dataset_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=test_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function
                )
            except Exception:
                dataset_test = BalancedDataset(
                    dataset_path,
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=test_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function
                )

        print(dataset_test)
        print("Dataset size =", len(dataset_test))

        # dataloader
    #     sampler_test = torch.utils.data.RandomSampler(dataset_test)
    #
    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test,
    #         sampler=sampler_test,
    #         batch_size=args.test_batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False,
    #     )
    #
    #     # ---- load checkpoints ----
    #     chkpt_files = [
    #         c for c in os.listdir(args.checkpoint_path)
    #         if c.endswith(".pth") and '-' in c
    #     ]
    #
    #     chkpt_pairs = []
    #     for c in chkpt_files:
    #         try:
    #             ep = int(c.split('-')[1].split('.')[0])
    #             chkpt_pairs.append((ep, c))
    #         except:
    #             continue
    #
    #     chkpt_pairs.sort(key=lambda x: x[0])
    #
    #     print("Checkpoint list:", chkpt_pairs)
    #
    #     for ep, ckpt_name in chkpt_pairs:
    #         ckpt_path = os.path.join(args.checkpoint_path, ckpt_name)
    #         print(f"\nLoading checkpoint: {ckpt_path}")
    #
    #         ckpt = torch.load(ckpt_path, map_location=device)
    #         target = model if not hasattr(model, "module") else model.module
    #         target.load_state_dict(ckpt['model'], strict=False)
    #
    #         test_stats = test_one_epoch(
    #             model=model,
    #             data_loader=data_loader_test,
    #             evaluator_list=evaluator_list,
    #             device=device,
    #             epoch=ep,
    #             log_writer=log_writer,
    #             args=args
    #         )
    #
    #         log_stats = {
    #             **{f'test_{k}': v for k, v in test_stats.items()},
    #             'epoch': ep
    #         }
    #
    #         # write log
    #         with open(os.path.join(args.full_log_dir, "log.txt"),
    #                   "a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")
    #
    # total_time = time.time() - start_time
    # print("Total testing time:", datetime.timedelta(seconds=int(total_time)))
    # print("\n=== DONE! ===")

        sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        # =====================================================
        # 🔥 只测试一个指定的 checkpoint，不遍历目录
        # =====================================================
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"❌ checkpoint_path 不是文件: {args.checkpoint_path}")

        chkpt_pairs = [(0, args.checkpoint_path)]  # epoch 强制设为 0

        print("Checkpoint list:", chkpt_pairs)

        # =====================================================
        # 🔥 加载并测试唯一一个 checkpoint
        # =====================================================
        for ep, ckpt_name in chkpt_pairs:
            ckpt_path = ckpt_name  # 不再拼接路径
            print(f"\nLoading checkpoint: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location=device)
            target = model if not hasattr(model, "module") else model.module
            target.load_state_dict(ckpt['model'], strict=False)

            test_stats = test_one_epoch(
                model=model,
                data_loader=data_loader_test,
                evaluator_list=evaluator_list,
                device=device,
                epoch=ep,
                log_writer=log_writer,
                args=args
            )

            log_stats = {
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': ep
            }

            with open(os.path.join(args.full_log_dir, "log.txt"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    print("Total testing time:", datetime.timedelta(seconds=int(total_time)))
    print("\n=== DONE! ===")


if __name__ == '__main__':
    args, model_args = get_args_parser()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
