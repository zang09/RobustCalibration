iter=(1)
# type=("from_lidar" "near" "far")
type=("far")

python3 geometry/main.py --scene_type kitti -s ~/data/RobustCalibration/5-0-t -m outputs/kitti/5-0-t/base

for i in ${!iter[@]}; do
    for t in ${type[@]}; do
        echo "Running experiment exp${iter[i]}_${t}"
        if [ ! -d outputs/kitti/5-0-t/exp${iter[i]}_${t} ]; then
            cp -r outputs/kitti/5-0-t/base outputs/kitti/5-0-t/exp${iter[i]}_${t}
        fi
        sleep 3
        python calibration/main.py -s ~/data/RobustCalibration/5-0-t -m outputs/kitti/5-0-t/exp${iter[i]}_${t} --init_method ${t} --render --loader kitti
    done
done