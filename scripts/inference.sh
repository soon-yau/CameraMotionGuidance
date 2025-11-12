export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=$1 
OUTPUT_PATH=outputs
CONFIG_FILE=${MODEL_PATH}/config.py
CAMERA_REF=eval/MotionCtrl/camera_poses/eval/test_camera_088b93f15ca8745d.txt
PROMPT_FILE=eval/simple_prompts.txt
SUBDIR='test'
POS_PROMPT='drone footage. high image quality.'

# loop through CFG and CMG scales
CFG_SCALE_T_LIST=("4") # CFG = 4
CFG_SCALE_C_LIST=("4") # CMG = 4

for CFG_SCALE_T in "${CFG_SCALE_T_LIST[@]}"
do
    for CFG_SCALE_C in "${CFG_SCALE_C_LIST[@]}"
    do
        CFG_SCALE_T_STR=$(echo "$CFG_SCALE_T" | tr '.' '-')
        CFG_SCALE_C_STR=$(echo "$CFG_SCALE_C" | tr '.' '-')
        CFG=cfg_${CFG_SCALE_T_STR}_${CFG_SCALE_C_STR}
        SAVE_DIR=${OUTPUT_PATH}/${SUBDIR}/${CFG} 

        torchrun --standalone --nproc_per_node 1 scripts/inference_camera.py ${CONFIG_FILE} \
        --disable-cache True \
        --ckpt-path $MODEL_PATH/model.pt  \
        --camera-ckpt-path $MODEL_PATH/camera.pt  \
        --camera-path $CAMERA_REF \
        --save-dir $SAVE_DIR \
        --batch-size 5 \
        --cfg-scale-t $CFG_SCALE_T \
        --cfg-scale-c $CFG_SCALE_C \
        --prompt-path $PROMPT_FILE \
        --pos-prompt "${POS_PROMPT}" 
    done
done