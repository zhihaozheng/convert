import os
import multiprocessing as mp

import click
import cv2
import numpy as np
import pathos.pools
from tqdm import tqdm

from cloudfiles import CloudFiles
from cloudvolume.lib import mkdir, touch

from temu.funs import get_pairs,get_good_pairs,get_region

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

    # progress_dir = mkdir(os.path.join(destination, 'progress_tif'))
    # done_files = set(os.listdir(progress_dir))
    to_download = list(all_files)
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
@click.argument("acq")
@click.argument("img")
@click.argument("pos_path")
@click.argument("save_path")
def getpairs(acq,img,pos_path,save_path):
    """
    pos_path: e.g. /media/voxa/WD_23/zhihao/ca3/tape3_blade2/211222/bladeseq-2021.12.24-14.55.36/s108-2021.12.24-14.55.36/metadata/stage_positions.csv
    tile_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s108-2021.12.24-14.55.36
    save_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/lst
    """
    get_pairs(acq,img,pos_path,save_path)

@main.command()
@click.argument("acq")
@click.argument("tile_path")
@click.argument("pos_path")
@click.argument("save_path")
@click.option('--exclude', type=str, default="", help="path to a file containing a lst of tiles to exclude", show_default=True)
def getgoodpairs(acq,tile_path,pos_path,save_path,exclude):
    '''
    acq_path: e.g. /media/voxa/WD_23/zhihao/ca3/tape3_blade2/211222/bladeseq-2021.12.24-14.55.36/s108-2021.12.24-14.55.36
    tile_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s108-2021.12.24-14.55.36
    save_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps
    exclude: e.g.

    !! save_path assume dir structure like this
    save_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/"
    <save_path>/maps/{}/summary.out
    summary_f=os.path.join(save_path,maps,acq,summary.out)
    lst_save_path = os.path.join(save_path,lst)
    acq = "s042"
    tile_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s042-2021.12.07-23.46.41"
    summary_f = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/maps/{}/summary.out".format(acq)
    pos_path = "/media/voxa/WD_23/zhihao/ca3/tape3_blade2/211208/bladeseq-2021.12.07-23.46.41/s042-2021.12.07-23.46.41/metadata/stage_positions.csv"
    save_path = "/media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps/lst/"
    '''
    # pos_path = os.path.join(acq_path,"metadata","stage_positions.csv")
    # acq = acq_path.split("/")[-1].split("-")[0]
    summary_f=os.path.join(save_path,"maps",acq,"summary.out")
    lst_save_path = os.path.join(save_path,"lst")
    if len(exclude)==0:
        exclude=[]
    else:
        with open(exclude,"r") as f:
            exclude=f.read().splitlines()

    get_good_pairs(acq,summary_f,tile_path,pos_path,lst_save_path,exclude,fname="core",corr_threshold=0.85)

# example script: temu getscript --img --acq --output --rst
@main.command()
@click.option('--img', default="", required=True)
@click.option('--acq', default="", required=True, help="acq, e.g. s010")
@click.option('--output', default="scripts.sh", required=True)
@click.option('--rst', default=False, is_flag=True, help="")
@click.option('--register', default=False, is_flag=True, help="")
@click.option('--align', default=False, is_flag=True, help="")
@click.option('--imap', default=False, is_flag=True, help="")
@click.option('--apply_map_red', default=None, type=str)
@click.option('--apply_map_hres', default=None, help="apply_map_dir")
@click.option('--size', default=None, help="used for apply_map_hres only")
@click.option('--mpi', default=22, type=int, help="number of parallel processes for mpirun")
@click.option('--serial/--no-serial', default=True)
def getscript(img, acq, output, rst, register, align, imap, apply_map_red, apply_map_hres, size, mpi, serial):
    '''
    --apply_map_red: "output path, will append acq subdir"
      produce alignTK script to run stitching
    --img : directory to tif images, wo the last slash
            e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s106-2021.12.22-14.51.58
    --acq: "acq, e.g. s010"
    --output: "directory where alignTK script will be saved to"
    --serial:
        if it's non serial
         append a list of img dir, acqs, one at a time
    '''
    if serial:
        txt = gen_cmd(img, acq, rst, register, align, imap, apply_map_red, apply_map_hres, size, mpi)
    else:
        txt_lst = []
        # read the list
        with open(img,"r") as f:
            imgs=f.read().splitlines()
        with open(img,"r") as f:
            acqs=f.read().splitlines()
        for i in range(len(acqs)):
            txt_lst.append(gen_cmd(imgs[i], acq[i], rst, register, align, imap, apply_map_red, apply_map_hres, size, mpi))
        txt = "".join(txt_lst)
    with open(output,"w") as f:
        f.write(txt)

def gen_cmd(img, acq, rst, register, align, imap, apply_map_red, apply_map_hres, size, mpi):
    txt_lst = []
    if rst:
        # acq, img
        txt_lst.append("mpirun -np {p} find_rst -pairs lst/{acq}_pairs.lst -tif -images {img}/ -output cmaps/{acq}/ -max_res 2048 -scale 1.0 -summary cmaps/{acq}/summary.out -margin 6 -rotation 0 -tx -100-100 -ty -100-100 -trans_feature 8 -distortion 1.0;".format(acq=acq,img=img,p=mpi))
    if register:
        txt_lst.append("mpirun -np {p} register -pairs lst/{acq}_pairs.lst -images {img}/ -output maps/{acq}/ -initial_map cmaps/{acq}/ -distortion 13.0 -output_level 7 -depth 6 -quality 0.1 -summary maps/{acq}/summary.out -min_overlap 10.0;".format(acq=acq,img=img,p=mpi))
    if align:
        txt_lst.append("mpirun -np {p} align -images {img}/ -image_list lst/{acq}_core_images.lst -maps maps/{acq}/ -map_list lst/{acq}_core_pairs.lst -output amaps/{acq}/ -schedule schedule_1.lst -incremental -output_grid grids/{acq}/ -grid_size 8192x8192 -fold_recovery 360;".format(acq=acq,img=img,p=mpi))
    if apply_map_red:
    # acq, img, output
        txt_lst.append("apply_map -image_list lst/{acq}_core_images.lst -images {img}/ -maps amaps/{acq}/ -output {odir}/{acq}/ -memory 7000 -overlay -rotation -30 -rotation_center 20000,0 --reduction 16;".format(acq=acq,img=img, odir=apply_map_red))
    if imap:
    # acq, img
        txt_lst.append("gen_imaps -image_list lst/{acq}_core_images.lst -images {img}/ -map_list lst/{acq}_core_pairs.lst -output imaps/{acq}/ -maps maps/{acq}/;".format(acq=acq,img=img))
    if apply_map_hres:
        txt_lst.append("apply_map -image_list lst/{acq}_core_images.lst -images {img}/ -maps amaps/{acq}/ -output {odir}/ -memory 7000 -overlay -rotation -30 -rotation_center 20000,0 -imaps imaps/{acq}/ -tile 2048x2048 -region ".format(acq=acq, img=img, odir=apply_map_hres) + funs.get_region(size)) + ";"
    return "".join(txt_lst)
