import pandas as pd
import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_pair_list(subt):
    pair_list = []
    for xi in pd.unique(subt.xis):
        xt = subt.query("xis==@xi")
        cl = [(i,j) for i in xt.yis for j in xt.yis if j-i==1]
        for pair in cl:
            tile_pair = xt.query("yis in @pair")["fname"]
            pair_list.append(" ".join(tile_pair) + " " + "-".join(tile_pair))

    for yi in pd.unique(subt.yis):
        yt = subt.query("yis==@yi")
        cl = [(i,j) for i in yt.xis for j in yt.xis if j-i==1]
        for pair in cl:
            tile_pair = yt.query("xis in @pair")["fname"]
            pair_list.append(" ".join(tile_pair) + " " + "-".join(tile_pair))
    return pair_list

def get_subtile_loc(pos_path, tile_path):
    sp = pd.read_csv(pos_path, skipinitialspace=True)
    delta = 44395.385
    x_min = min(sp.stage_x_nm)
    y_max = max(sp.stage_y_nm)
    sp = sp.assign(xi=round((sp.stage_x_nm - x_min)/delta),
    yi=round( (y_max - sp.stage_y_nm)/delta )).astype({"xi": int, "yi": int})

    x_map = {6:0,7:1,8:2,5:0,0:1,1:2,4:0,3:1,2:2}
    y_map = {6:0,5:1,4:2,7:0,0:1,3:2,8:0,1:1,2:2}

    # for all files, create rows and columns
    fnames = []
    xis = []
    yis = []
    supertiles = []
    subtiles = []
    for fname in os.listdir(tile_path):
        t1 = int(fname.split("_")[1])
        t2 = int(fname.split("_")[2].split(".")[0])
        supertiles.append(t1)
        subtiles.append(t2)
        fnames.append(fname.split(".")[0])
        xis.append(sp.loc[t1,"xi"]*3 + x_map[t2])
        yis.append(sp.loc[t1,"yi"]*3 + y_map[t2])

    return pd.DataFrame({"fname":fnames,"xis":xis,"yis":yis,
    "supertile":supertiles,"subtile":subtiles})
# subtile_loc.to_csv(path + "subtile_positions.csv")

def get_align_cmd(acq):
    t1 = "mpirun -np 29 find_rst -pairs lst/s{acq}_pairs.lst -tif -images data/tif/s{acq}/ -output cmaps/s{acq}/ -max_res 2048 -scale 1.0 -summary cmaps/s{acq}/summary.out -margin 6 -rotation 0 -tx -100-100 -ty -100-100 -trans_feature 8 -distortion 1.0;".format(acq=acq)
    t2 = "mpirun -np 29 register -pairs lst/s{acq}_pairs.lst -images data/tif/s{acq}/ -output maps/s{acq}/ -initial_map cmaps/s{acq}/ -distortion 13.0 -output_level 7 -depth 6 -quality 0.1 -summary maps/s{acq}/summary.out -min_overlap 10.0;".format(acq=acq)
    t3 = "mpirun -np 29 align -images data/tif/s{acq}/ -image_list lst/s{acq}_images.lst -maps maps/s{acq}/ -map_list lst/s{acq}_pairs.lst -output amaps/s{acq}/ -schedule schedule_1.lst -incremental -output_grid grids/s{acq}/ -grid_size 4096x4096 -fold_recovery 360;".format(acq=acq)
    return "\n".join([t1,t2,t3])

def apply_map_cmd_initial(a):
    return "apply_map -image_list lst/s{acq}_images.lst -images data/tif/s{acq}/ -maps amaps/s{acq}/ -output aligned/s{acq}/ -memory 7000 -overlay -reduction 4".format(acq=a)

def get_tile_size(n):
    for i in range(5):
        j = np.ceil(n/1000 + 0.5)+ i
        if j % 4 == 0:
            return int(j*1000)

def get_region(path="/home/voxa/Documents/zhihao/211228-small_stack_stitch/aligned/"):
    '''
    path: path to the size file
    '''
    # size_f = os.path.join(path,"s" + i,"size")
    with open(path,"r") as f:
        contents = f.readlines()
    nums = [int(i) for i in re.search(r'(\d+)x(\d+)-(\d+)-(\d+)\n',contents[1]).groups()]
    wid = int(np.ceil(nums[0]/2048)*2048)
    hei = int(np.ceil(nums[1]/2048)*2048)
    x1 = int((wid - nums[0])//2 + nums[2])
    y1 = int((hei - nums[1])//2 + nums[3])
    return '{}x{}-{}-{}'.format(wid,hei,x1,y1)

def get_good_pairs(acq_label,summary_f,tile_path,pos_path,save_path,exclude=[],fname="core"):
    '''
    acq_label: label for the section that will be attached to the pair list name
    summary_f: path to the summary stats after register
    tile_path: path to the tiles that will be stitched
    pos_path: path to the stage_position.csv (stage_positions.csv)
    save_path: path to where the pair_list and image_list will save to
    exclude: tiles to be excluded and any tile connected to it will be excluded either
    '''
    # open the file and figure out where the first table ends
    a_file = open(summary_f,"r")
    phrase = "Sorted by energy:"
    for number, line in enumerate(a_file):
      if phrase in line:
        line_number = number
        break
    a_file.close()

    # read the table with correlation numbers
    summary = pd.read_csv(summary_f, header=1,sep=" ",nrows=line_number-4, skipinitialspace=True)
    lowcorr = summary.query("CORRELATION<0.85")[["IMAGE","REFERENCE"]]
    lowcorr_pairs = [set(lowcorr.iloc[i]) for i in range(len(lowcorr))]
    align_pair_list = []
    align_image_list = []

    # pos_path = os.path.join(data_path,acq_name,"metadata","stage_positions.csv")
    subt = get_subtile_loc(pos_path, tile_path)
    pair_list = get_pair_list(subt)
    for pair in pair_list:
        ps = set(pair.split(" ")[:2])
        ps_list = list(ps)
        if ps in lowcorr_pairs:
            continue
        elif len(exclude)>0 and (ps_list[0] in exclude or ps_list[1] in exclude):
            continue
        else:
            align_pair_list.append(pair)
            align_image_list.extend(ps_list)
    align_image_list = list(set(align_image_list))

    G = nx.parse_edgelist([i[:23] for i in align_pair_list])
    lst = sorted(nx.connected_components(G), key=len, reverse=True)[1:]
    fls = set().union(*lst)
    core_align_pairs = [i for i in align_pair_list
                        if i[:11] not in fls and i[13:23] not in fls]
    core_images = [i for i in align_image_list if i not in fls]
    with open(os.path.join(save_path,acq_label + "_" + fname + "_pairs.lst"),"w") as f:
        f.write("\n".join(core_align_pairs))
    with open(os.path.join(save_path,acq_label + "_" + fname + "_images.lst"),"w") as f:
        f.write("\n".join(core_images))
    print("pairs_images_lst saved")



def get_pairs(acq_label,tile_path,pos_path,save_path):
    '''
    acq_label: label for the section that will be attached to the pair list name
    tile_path: path to the tiles that will be stitched
    pos_path: path to the stage_position.csv (stage_positions.csv)
    save_path: path to where the pair_list and image_list will save to
    '''
    subt = get_subtile_loc(pos_path, tile_path)
    pair_list = get_pair_list(subt)

    with open(save_path + acq_label + "_pairs.lst","w") as f:
        f.write("\n".join(pair_list))
    with open(save_path + acq_label + "_images.lst","w") as f:
        f.write("\n".join(subt.fname))
    print(acq_label + " is done")

def apply_map_cmd(acq):
    return "apply_map -image_list lst/s{acq}_images.lst -images data/tif/s{acq}/ -maps amaps/s{acq}/ -output aligned/s{acq}/ -memory 7000 -overlay -tile 2048x2048 -region ".format(acq=acq) + get_region(acq)

def get_upload_cmd(acq):
  return "tem2ng -p 12 upload aligned/s{acq}/4k_native/ tigerdata://sseung-archive/pni-tem1/cutout94_native/ --z {acq}".format(acq=acq)

def ls(t1,t2=4): return list(range(t1,t1+t2))

def get_ls(initials,step=4): return [item for j in [ls(i,step) for i in initials] for item in j]


def plot_stage_positions(pos_path,save_path,acq_label):
    pos = pd.read_csv(pos_path, header=0, skipinitialspace=True)
    fig, ax = plt.subplots()
    ax.scatter(pos.stage_x_nm,pos.stage_y_nm)
    for i in range(pos.shape[0]):
        ax.annotate(pos.loc[i,"tile_id"],(pos.loc[i,"stage_x_nm"],pos.loc[i,"stage_y_nm"]))
    # plt.title(d)
    fig.set_size_inches(12,8)
    plt.savefig(save_path + acq_label)
    plt.close()
