import os
import multiprocessing as mp

import click
import cv2
import numpy as np
import pathos.pools
from tqdm import tqdm

from cloudfiles import CloudFiles




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
def to_tif(ctx, source, destination):
    """.
    """
    cf = CloudFiles(source, progress=True)


    vol = CloudVolume(destination)
    progress_dir = mkdir(os.path.join(source, 'progress'))

    done_files = set(os.listdir(progress_dir))
    all_files = os.listdir(source)
    all_files = set([
        fname for fname in all_files
        if (
            os.path.isfile(os.path.join(source, fname))
            and os.path.splitext(fname)[1] == ".bmp"
        )
    ])
    to_upload = list(all_files.difference(done_files))
    to_upload.sort()

    def process(filename):

        img = cv2.imread(os.path.join(source, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.transpose(img)
        while img.ndim < 4:
            img = img[..., np.newaxis]

        bbx = Bbox.from_filename(get_ng(filename, z=z))
        vol[bbx] = img
        touch(os.path.join(progress_dir, filename))
        return 1

    parallel = int(ctx.obj.get("parallel", 1))

    with tqdm(desc="Upload", total=len(all_files), initial=len(done_files)) as pbar:
        with pathos.pools.ProcessPool(parallel) as pool:
            for num_inserted in pool.imap(process, to_upload):
                pbar.update(num_inserted)
