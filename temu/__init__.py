import os
import multiprocessing as mp

import click
import cv2
import numpy as np
import pathos.pools
from tqdm import tqdm

from cloudfiles import CloudFiles
from cloudvolume.lib import mkdir, touch

from temu.funs import get_pairs

@click.group()
@click.option("-p", "--parallel", default=1, help="Run with this number of parallel processes. If 0, use number of cores.")
@click.pass_context
def main(ctx, parallel):
    """
    """
    parallel = int(parallel)
    if parallel == 0:
        parallel = mp.cpu_count()
    ctx.ensure_object(dict)
    ctx.obj["parallel"] = max(min(parallel, mp.cpu_count()), 1)

@main.command()
@click.argument("source")
@click.argument("destination")
@click.pass_context
def totif(ctx, source, destination):
    tile_path = os.path.join(source,"subtiles")

    target_path = os.path.join(destination, source.split("/")[-1])
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    progress_dir = mkdir(os.path.join(target_path, 'progress_tif'))
    done_files = set(os.listdir(progress_dir))
    all_files = os.listdir(tile_path)

    all_files = set([
            fname for fname in all_files
            if os.path.splitext(fname)[1] == ".bmp"
        ])

    to_cv = list(all_files.difference(done_files))
    to_cv.sort()

    def process(filename):
        img = cv2.imread(os.path.join(tile_path,filename),cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(target_path, filename.replace("bmp","tif")), img)
        return 1

    parallel = int(ctx.obj.get("parallel", 1))

    with tqdm(desc="Convert", total=len(all_files), initial=len(done_files)) as pbar:
        with pathos.pools.ProcessPool(parallel) as pool:
            for num_inserted in pool.imap(process, to_cv):
                pbar.update(num_inserted)


@main.command()
@click.argument("source")
@click.argument("destination")
@click.pass_context
def downloadtif(ctx, source, destination):
    """
    source:
    """
    cf = CloudFiles(source, progress=True)

    all_files = set([
        fname for fname in cf.list(prefix="tile_", flat=True)
        if ".bmp" in fname
        ])

    if not os.path.exists(destination):
        os.makedirs(destination)

    progress_dir = mkdir(os.path.join(destination, 'progress_tif'))
    done_files = set(os.listdir(progress_dir))
    to_download = list(all_files.difference(done_files))
    to_download.sort()

    def process(filename):
        img = cf.get(filename)
        bmp = cv2.imdecode(np.frombuffer(img,dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(destination, filename.replace("bmp","tif")), bmp)
        return 1

    parallel = int(ctx.obj.get("parallel", 1))

    with tqdm(desc="download", total=len(all_files)) as pbar:
        with pathos.pools.ProcessPool(parallel) as pool:
            for num_inserted in pool.imap(process, to_download):
                pbar.update(num_inserted)

@main.command()
@click.argument("acq_path")
@click.argument("tile_path")
@click.argument("save_path")
def getpairs(acq_path,tile_path,save_path):
    """
    acq_path: e.g. /media/voxa/WD_23/zhihao/ca3/tape3_blade2/211222/bladeseq-2021.12.24-14.55.36/s108-2021.12.24-14.55.36
    tile_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s108-2021.12.24-14.55.36
    save_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/lst
    """
    pos_path = os.path.join(acq_path,"metadata","stage_positions.csv")
    acq_label = acq_path.split("/")[-1].split("-")[0]
    get_pairs(acq_label,tile_path,pos_path,save_path)


@main.command()
@click.argument("acq_label")
@click.argument("tile_path")
@click.argument("pos_path")
@click.argument("save_path")
def getgoodpairs(acq_label,tile_path,pos_path,save_path):
    '''
    acq_path: e.g. /media/voxa/WD_23/zhihao/ca3/tape3_blade2/211222/bladeseq-2021.12.24-14.55.36/s108-2021.12.24-14.55.36
    tile_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s108-2021.12.24-14.55.36
    save_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps

    !! save_path assume dir structure like this
    save_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/"
    <save_path>/maps/{}/summary.out
    summary_f=os.path.join(save_path,maps,acq_label,summary.out)
    lst_save_path = os.path.join(save_path,lst)
    acq_label = "s042"
    tile_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s042-2021.12.07-23.46.41"
    summary_f = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/maps/{}/summary.out".format(acq_label)
    pos_path = "/media/voxa/WD_23/zhihao/ca3/tape3_blade2/211208/bladeseq-2021.12.07-23.46.41/s042-2021.12.07-23.46.41/metadata/stage_positions.csv"
    save_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/lst/"
    '''
    # pos_path = os.path.join(acq_path,"metadata","stage_positions.csv")
    # acq_label = acq_path.split("/")[-1].split("-")[0]
    summary_f=os.path.join(save_path,"maps",acq_label,"summary.out")
    lst_save_path = os.path.join(save_path,"lst")
    get_good_pairs(acq_label,summary_f,tile_path,pos_path,lst_save_path,exclude=[],fname="core",corr_threshold=0.85)
