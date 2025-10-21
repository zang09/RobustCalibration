#!/bin/bash

iter=(1)

type=("from_lidar")
# type=("from_lidar" "near" "far")

cam_ids=(0 1 2 3)


python geometry/main.py --scene_type custom -s data/kitti360/straight -m outputs/kitti360/straight/base

for i in ${!iter[@]}; do
    for t in ${type[@]}; do
        for cam_id in ${cam_ids[@]}; do
            echo "Running experiment exp${iter[i]}_${t}_cam${cam_id}"
            if [ ! -d outputs/kitti360/straight/exp${iter[i]}_${t}_cam${cam_id} ]; then
                cp -r outputs/kitti360/straight/base outputs/kitti360/straight/exp${iter[i]}_${t}_cam${cam_id}
            fi
            sleep 3
            python calibration/main.py -s data/kitti360/straight -m outputs/kitti360/straight/exp${iter[i]}_${t}_cam${cam_id} --init_method ${t} --cam_id ${cam_id} --render --loader custom
        done
    done
done