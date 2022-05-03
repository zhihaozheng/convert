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
@click.option('--exclude', type=str, default="", help="path to a file containing a lst of tiles to exclude", show_default=True)
def getgoodpairs(acq_label,tile_path,pos_path,save_path,exclude):
    '''
    acq_path: e.g. /media/voxa/WD_23/zhihao/ca3/tape3_blade2/211222/bladeseq-2021.12.24-14.55.36/s108-2021.12.24-14.55.36
    tile_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s108-2021.12.24-14.55.36
    save_path: e.g. /media/voxa/WD_36/zhihao/ca3/tape3_blade2_maps
    exclude: e.g.

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
    if len(exclude)==0:
        exclude=[]
    else:
        with open(exclude,"r") as f:
            exclude=f.read().splitlines()

    get_good_pairs(acq_label,summary_f,tile_path,pos_path,lst_save_path,exclude,fname="core",corr_threshold=0.85)

# example script: temu tk --img_dir --acq_label --output_dir get_tk --rst
@main.group("tk")
@click.argument('--img_dir', default="", required=True, help="directory to tif images, wo the last slash")
@click.argument('--acq_label', default="", required=True, help="acq_label, e.g. s010")
@click.argument('--output_dir', default="scripts.sh", help="directory where alignTK script will be saved to")
def tkgroup(ctx,img_dir, acq_label, output_dir):
  """
  produce alignTK script to run stitching
  --img_dir : /media/voxa/WD_36/zhihao/ca3/tape3_blade2_tif/s106-2021.12.22-14.51.58

  """
  ctx.ensure_object(dict)
  ctx.obj["img_dir"] = img_dir
  ctx.obj["acq_label"] = acq_label
  ctx.obj["output_dir"]= output_dir

@tkgroup.command()
@click.option('--rst', default=False, is_flag=True, help="")
@click.option('--register', default=False, is_flag=True, help="")
@click.option('--align', default=False, is_flag=True, help="")
@click.option('--imap', default=False, is_flag=True, help="")
@click.argument('--apply_map_red', default=None, type=str, help="output path, will append acq_label subdir")
@click.pass_context
def get_script(ctx, rst, register, align, imap, apply_map_red):
    img_dir = ctx.object.get("img_dir")
    acq = ctx.object.get("acq_label")
    output_dir = ctx.object.get("output_dir")

    txt_lst = []
    if rst:
        # acq, img_dir
        txt_lst.append("mpirun -np 20 find_rst -pairs lst/{acq}_pairs.lst -tif -images {img_dir}/ -output cmaps/{acq}/ -max_res 2048 -scale 1.0 -summary cmaps/{acq}/summary.out -margin 6 -rotation 0 -tx -100-100 -ty -100-100 -trans_feature 8 -distortion 1.0;".format(acq=acq,img_dir=img_dir))
    elif register:
        txt_lst.append("mpirun -np 20 register -pairs lst/s{acq}_pairs.lst -images {img_dir}/ -output maps/s{acq}/ -initial_map cmaps/s{acq}/ -distortion 13.0 -output_level 7 -depth 6 -quality 0.1 -summary maps/s{acq}/summary.out -min_overlap 10.0;".format(acq=acq,img_dir=img_dir))
    elif align:
        txt_lst.append("mpirun -np 22 align -images {img_dir}/ -image_list lst/{acq}_core_images.lst -maps maps/{acq}/ -map_list lst/{acq}_core_pairs.lst -output amaps/{acq}/ -schedule schedule_1.lst -incremental -output_grid grids/{acq}/ -grid_size 8192x8192 -fold_recovery 360;".format(acq=acq,img_dir=img_dir))
    elif apply_map_red:
    # acq, img_dir, output_dir
        txt_lst.append("apply_map -image_list lst/{acq}_core_images.lst -images {img_dir}/ -maps amaps/{acq}/ -output {odir}/{acq}/ -memory 7000 -overlay -rotation -30 -rotation_center 20000,0 --reduction 16;".format(acq=acq,img_dir=img_dir, odir=apply_map_red))
    elif imap:
    # acq, img_dir
        txt_lst.append("gen_imaps -image_list lst/{acq}_core_images.lst -images {img_dir}/ -map_list lst/{acq}_core_pairs.lst -output imaps/{acq}/ -maps maps/{acq}/;".format(acq=acq,img_dir=img_dir))
    with open(output_dir,"w") as f:
        f.write("".join(txt_lst))


@tkgroup.command()
@click.argument('size_path')
@click.argument('apply_map_dir')
@click.pass_context
def get_apply_map_fullres(ctx,):
    img_dir = ctx.object.get("img_dir")
    acq = ctx.object.get("acq_label")
    output_dir = ctx.object.get("output_dir")
    # acq, img_dir, size_path,
    with open(output_dir,"w") as f:
        f.write(
        "apply_map -image_list lst/{acq}_core_images.lst -images {img_dir}/ -maps amaps/{acq}/ -output {odir} -memory 7000 -overlay -rotation -30 -rotation_center 20000,0 -imaps imaps/{acq}/ -tile 2048x2048 -region ".format(acq=acq, img_dir=img_dir, odir=apply_map_dir) + funs.get_region(size_path)
        )
