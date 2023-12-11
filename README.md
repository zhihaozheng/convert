
*work_dir* is expected to have the following file structure for metadata and input, to run alignTK. The map directories have the same structure as in *amaps*
```
(working directory)
 |-- amaps
 |   |-- s0
 |   |-- s1
 |   |-- s2
 |   |-- ...
 |
 |-- cmaps
 |-- grids
 |-- imaps
 |-- lst
 |-- maps


 |-- schedule_2.lst
 |-- slurm_scripts
 |-- stage_pos


In addition, 
a directory that points to the tiff images is needed (*--tile_path*)
an input that list the sections to be processed (*--acqs*)