now=$(date +"%Y%m%d_%H%M%S")
dataset_name="carla"
exp_name="$dataset_name-$now"


export CUDA_VISIBLE_DEVICES=4
export CUDA_LAUNCH_BLOCKING=1

# python test_mutual_cldm_seq.py
python test_multiple_diffusion.py --exp_name ${exp_name} --window_size 5
# python test_single_seq.py