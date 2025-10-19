iter=(1)
# type=("from_lidar" "near" "far")
type=("far")

# python3 geometry/main.py --scene_dir 5-0-t -m output/5-0-t/base

for i in ${!iter[@]}; do
    for t in ${type[@]}; do
        echo "Running experiment exp${iter[i]}_${t}"
        cp -r output/5-0-t/base output/5-0-t/exp${iter[i]}_${t}
        sleep 3
        python calibration/main.py -s 5-0-t -m output/5-0-t/exp${iter[i]}_${t} --init_method ${t} --render
    done
done