# Perception in robotics. Final project





How to use:



1. Clone this https://github.com/magicleap/SuperGluePretrainedNetwork inside this repo , install requirements 
2. Download data, for example from here: https://disk.yandex.ru/d/kVjlNLCaokkfEw
3. Create necessary folder (./results), run this:

`python metrics.py --path_to_pairs ./results/kitti_test_pairs.txt --path_to_images data/kitti/campus/2011_09_28_image/image_00/data/ --path_to_output ./results/kitti_test_output --viz`

(you can delete --viz if you don't need it)

