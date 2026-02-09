import os
import json
import time
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.evaluation import PixelF1, ImageF1
from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch

from conv_cswin import Mesorch_ConvNeXt_CSWinB


def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)

    parser.add_argument('--model', default="Mesorch", type=str,
                        help='The name of applied model', required=True)

    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input')

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--if_padding', action='store_true')
    parser.add_argument('--if_resizing', action='store_true')

    parser.add_argument('--edge_mask_width', default=None, type=int)

    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str)
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--no_model_eval', action='store_true')
    parser.add_argument('--test_period', default=4, type=int)

    parser.add_argument('--log_per_epoch_count', default=20, type=int)
    parser.add_argument('--find_unused_parameters', action='store_true')

    parser.add_argument('--if_not_amp', action='store_false')
    parser.add_argument('--accum_iter', default=16, type=int)

    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=4)

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    args, remaining_args = parser.parse_known_args()

    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args


def main(args, model_args):
    args.distributed = False
    args.world_size = 1
    args.local_rank = 0
    global_rank = 0

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed
    misc.seed_torch(seed)
    np.random.seed(seed)

    train_transform = get_albu_transforms('train')
    test_transform = get_albu_transforms('test')

    post_function_name = f"{args.model}_post_func".lower()
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None

    if os.path.isdir(args.data_path):
        dataset_train = ManiDataset(
            args.data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=train_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    else:
        try:
            dataset_train = JsonDataset(
                args.data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        except:
            dataset_train = BalancedDataset(
                args.data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=train_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )

    if os.path.isdir(args.test_data_path):
        dataset_test = ManiDataset(
            args.test_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function
        )
    else:
        try:
            dataset_test = JsonDataset(
                args.test_data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        except:
            dataset_test = BalancedDataset(
                args.test_data_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model_fn = MODELS.get(args.model)

    if isinstance(model_fn, (types.FunctionType, types.MethodType)):
        model_params = inspect.signature(model_fn).parameters
    else:
        model_params = inspect.signature(model_fn.__init__).parameters

    combined_args = {k: v for k, v in vars(args).items() if k in model_params}
    for k, v in vars(model_args).items():
        if k in model_params and k not in combined_args:
            combined_args[k] = v

    model = model_fn(**combined_args)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    args.opt = 'AdamW'
    args.betas = (0.9, 0.999)
    args.momentum = 0.9

    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)

    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler
    )

    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
    ]

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_evaluate_metric_value = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args
        )

        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        optimizer.zero_grad()

        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            test_stats = test_one_epoch(
                model,
                data_loader=data_loader_test,
                evaluator_list=evaluator_list,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args
            )

            evaluate_metric = evaluator_list[0].name
            metric_value = test_stats[evaluate_metric]

            if metric_value > best_evaluate_metric_value:
                best_evaluate_metric_value = metric_value
                print(f"Best {evaluate_metric} = {best_evaluate_metric_value}")
                if epoch > 35:
                    misc.save_model(
                        args=args, model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        epoch=epoch
                    )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                'epoch': epoch
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                'epoch': epoch
            }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
