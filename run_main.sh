set -x
nohup python3 Apex_codes/main.py  \
                    --gpu_number='0,1' \
                    --batchSize=7  \
                    --lr=5e-4  \
                    --lr_decay_factor=0.5 \
                    --model_use='origin_UNet' \
                    --Epoch=5 \
                    --saveEpoch=1000000 \
                    --lr_decay_per_epoch=20 \
                    --data_dir='/home/user1/Documents/css-unet-main/UNET1/diamonds_labels_clipped' \
                    --val_id_loc='/home/user1/Documents/css-unet-main/UNET1/diamonds_labels_clipped/id_val.txt' \
                    --id_loc='/home/user1/Documents/css-unet-main/UNET1/id.txt'  \
                    --model_save_dir='/home/user1/Documents/css-unet-main/unet1_results/Models' \
                    --save_results_folder='/home/user1/Documents/css-unet-main/unet1_results/Results' \
                    --save_images_folder='/home/user1/Documents/css-unet-main/unet1_results/Images' \
                    --resume_training=0 \
                    > 0318_rangefilt.log 2>&1 &
set +x
