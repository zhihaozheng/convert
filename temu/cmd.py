
def get_align_cmd(acq):
    t1 = "mpirun -np 29 find_rst -pairs lst/s{acq}_pairs.lst -tif -images data/tif/s{acq}/ -output cmaps/s{acq}/ -max_res 2048 -scale 1.0 -summary cmaps/s{acq}/summary.out -margin 6 -rotation 0 -tx -100-100 -ty -100-100 -trans_feature 8 -distortion 1.0;".format(acq=acq)
    t2 = "mpirun -np 29 register -pairs lst/s{acq}_pairs.lst -images data/tif/s{acq}/ -output maps/s{acq}/ -initial_map cmaps/s{acq}/ -distortion 13.0 -output_level 7 -depth 6 -quality 0.1 -summary maps/s{acq}/summary.out -min_overlap 10.0;".format(acq=acq)
    t3 = "mpirun -np 29 align -images data/tif/s{acq}/ -image_list lst/s{acq}_images.lst -maps maps/s{acq}/ -map_list lst/s{acq}_pairs.lst -output amaps/s{acq}/ -schedule schedule_1.lst -incremental -output_grid grids/s{acq}/ -grid_size 4096x4096 -fold_recovery 360;".format(acq=acq)
    return "\n".join([t1,t2,t3])

def apply_map_cmd_initial(a):
    return "apply_map -image_list lst/s{acq}_images.lst -images data/tif/s{acq}/ -maps amaps/s{acq}/ -output aligned/s{acq}/ -memory 7000 -overlay -reduction 4".format(acq=a)

def apply_map_cmd(acq):
    return "apply_map -image_list lst/s{acq}_images.lst -images data/tif/s{acq}/ -maps amaps/s{acq}/ -output aligned/s{acq}/ -memory 7000 -overlay -tile 2048x2048 -region ".format(acq=acq) + get_region(acq)

def get_upload_cmd(acq):
  return "tem2ng -p 12 upload aligned/s{acq}/4k_native/ tigerdata://sseung-archive/pni-tem1/cutout94_native/ --z {acq}".format(acq=acq)
