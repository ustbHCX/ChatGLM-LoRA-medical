
CUDA_VISIBLE_DEVICES=0 python generate_data.py \
    --genrate_data_dir='./genrate_data_dir'\
    --target_dir_list=['./chinese_dataset/其他中文问题补充/',
                   './chinese_dataset/翻译后的中文数据/',
                   './chinese_dataset/医学领域数据/',]