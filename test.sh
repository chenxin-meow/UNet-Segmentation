
set -x
python3 Apex_codes/main.py   --gpu_number='0' \
                    --batchSize=1  \
                    --lr=1e-4  \
                    --train_or_test='test'  \
                    --data_dir='/home/user1/Documents/css-unet-main/UNET1/diamonds_one'  \
                    --id_loc='/home/user1/Documents/css-unet-main/UNET1/diamonds_one/id_test.txt'  \
                    --save_images_folder='/home/user1/Documents/css-unet-main/unet1_results/Images-one'  \
                    --model_load_dir='/home/user1/Documents/css-unet-main/unet1_results/Models/2021_01_27-batchSize_7-Epoch_250-lr_0.0001-re_0-UNet_val_best'
set +x