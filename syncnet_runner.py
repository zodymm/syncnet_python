#!/usr/bin/python

# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy
import torch.multiprocessing as mp
from tqdm import tqdm

from SyncNetInstance import SyncNetInstance


def parse_args(data_dir, videofile, reference):
    opt = argparse.Namespace()
    setattr(opt, "data_dir", data_dir)
    setattr(opt, "videofile", videofile)
    setattr(opt, "reference", reference)

    setattr(opt, "batch_size", 20)
    setattr(opt, "vshift", 15)

    setattr(opt, "avi_dir", os.path.join(opt.data_dir, "pyavi"))
    setattr(opt, "tmp_dir", os.path.join(opt.data_dir, "pytmp"))
    setattr(opt, "work_dir", os.path.join(opt.data_dir, "pywork"))
    setattr(opt, "crop_dir", os.path.join(opt.data_dir, "pycrop"))

    return opt


def executor(opt, s):
    # ==================== LOAD MODEL AND FILE LIST ====================

    # print("Model %s loaded." % opt.initial_model)
    flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, "0*.avi"))
    flist.sort()

    # ==================== GET OFFSETS ====================

    dists = []
    minvals = []
    confs = []

    for idx, fname in enumerate(flist):
        try:
            offset, conf, dist, minval = s.evaluate(opt, videofile=fname)

        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")
            continue

        dists.append(dist)
        minvals.append(minval)
        confs.append(conf)

    # ==================== PRINT RESULTS TO FILE ====================

    with open(os.path.join(opt.work_dir, opt.reference, "activesd.pckl"), "wb") as fil:
        pickle.dump(dists, fil)

    with open(os.path.join(opt.work_dir, opt.reference, "res.txt"), "w") as f:
        f.write(f"LSE-D\t{numpy.mean(minvals, axis=0).astype(str).tolist()}")
        f.write("\n")
        f.write(f"LSE-C\t{numpy.mean(confs, axis=0).astype(str).tolist()}")
        f.write("\n")


def parse_results(data_dir):
    lse_ds = []
    lse_cs = []

    for res_text_path in Path(data_dir).rglob("res.txt"):
        lines = res_text_path.read_text().splitlines()

        if len(lines) < 2:
            print(f"[WARNING] Invalid content in: {res_text_path}")
            continue
        try:
            lse_d = float(lines[0].split("\t")[-1].strip())
            lse_c = float(lines[1].split("\t")[-1].strip())

            # Exclude NaN or non-finite values
            if numpy.isfinite(lse_d) and numpy.isfinite(lse_c):
                lse_ds.append(lse_d)
                lse_cs.append(lse_c)
            else:
                print(f"[WARNING] Non-finite value in: {res_text_path}")
        except Exception as e:
            print(f"[WARNING] Failed to parse {res_text_path}: {e}")

    if lse_ds and lse_cs:
        print(f"Mean LSE-D: {numpy.mean(lse_ds):.4f}")
        print(f"Mean LSE-C: {numpy.mean(lse_cs):.4f}")
    else:
        print("[ERROR] No valid data found.")


def main(data_dir, chunk, gpu_id, index):
    s = SyncNetInstance(device=gpu_id)
    s.loadParameters("data/syncnet_v2.model")

    pos = index
    pbar = tqdm(total=len(chunk), desc=f"Worker {index} ({gpu_id})", position=pos)

    for videofile in chunk:
        reference = Path(videofile).stem

        opt = parse_args(data_dir, videofile, reference)
        executor(opt, s)

        del opt

        pbar.update(1)

    pbar.close()
    pass


if __name__ == "__main__":
    argparse.ArgumentParser(description="Face Detection and Tracking")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/xxx")
    parser.add_argument("--video_dir", type=str, default="/path/to/video_files")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=3)

    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_workers = args.num_workers
    video_dir = args.video_dir
    output_base_dir = args.data_dir

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    video_files = list(Path(video_dir).rglob("*.mp4"))[:1000]

    chunks = numpy.array_split(video_files, len(gpu_ids) * num_workers)

    mp.set_start_method("spawn", force=True)
    processes = []
    for idx, chunk in enumerate(chunks):
        device = gpu_ids[idx % len(gpu_ids)]

        device = f"cuda:{device}"
        p = mp.Process(target=main, args=(output_base_dir, chunk, device, idx))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print("All syncnet processes finished.")
    print("Parsing results...")
    parse_results(output_base_dir)
