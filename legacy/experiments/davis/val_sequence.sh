cmd="
python s2_val_flow.py \
    --prefix new \
    --net sceneflow_field \
    --dataset davis_sequence \
    --track_id dog \
    --log_time \
    --vali_batches 150 \
    --batch_size 1 \
    --optim adam \
    --gpu 0 \
    --save_net 1 \
    --workers 4 \
    --acc_mul 1 \
    --disp_mul 1 \
    --warm_sf 5 \
    --scene_lr_mul 1000 \
    --repeat 1 \
    --flow_mul 1\
    --sf_mag_div 100 \
    --time_dependent \
    --gaps 1,2,3,4 \
    --midas \
    --use_disp \
    --logdir './checkpoints/davis/sequence/' \
    $*"
echo $cmd
eval $cmd



