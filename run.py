import logging
import argparse

import torch
import wandb
from pyhocon import ConfigFactory
from torch.torch_version import TorchVersion
from runners.train_runner import TrainRunner


if __name__ == "__main__":
    if torch.__version__ > TorchVersion("1.13.0+cu117"):
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARN)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    wdb_run = wandb.init(
        project="GARField",
        job_type=args.mode,
        notes=args.note,
        tags=[args.case] + args.tag.split(","),
        config=ConfigFactory.parse_file(args.conf).as_plain_ordered_dict(),
    )

    if args.mode == "train_render":

        runner = TrainRunner(args.conf, wdb_run, args.mode, args.case, args.is_continue)
        runner.train()
        runner.save_checkpoint(final=True)
        runner.save_pose_yaml(final=True)
        runner.render_image(wdb_run.name.replace("/", "-"))
        runner.render_scene_zslices(wdb_run.name.replace("/", "-"))
        print("Done")

    elif args.mode == "render_novel":

        runner = TrainRunner(
            args.conf,
            wdb_run,
            args.mode,
            args.case,
            args.is_continue,
            model_name=args.model,
        )
        runner.render_image("aug3")
        print("Done")

    elif args.mode == "render_slices":

        runner = TrainRunner(args.conf, wdb_run, args.mode, args.case, args.is_continue)
        runner.render_scene_zslices()
        print("Z slices - Done!")

    elif args.mode == "interp_view":

        runner = TrainRunner(
            args.conf,
            wdb_run,
            args.mode,
            args.case,
            args.is_continue,
            model_name=args.model,
        )
        runner.interpolate_view(0, 1, n_frames=200)
        print("View interpolation done")

    elif args.mode == "make_vid":

        runner = TrainRunner(
            args.conf,
            wdb_run,
            args.mode,
            args.case,
            args.is_continue,
            model_name=args.model,
        )
        runner.make_vid(0, 4)
        print("Vid generation done")

    if args.mode == "validate":

        runner = TrainRunner(
            args.conf,
            wdb_run,
            args.mode,
            args.case,
            args.is_continue,
            model_name=args.model,
        )
        runner.get_stats()
