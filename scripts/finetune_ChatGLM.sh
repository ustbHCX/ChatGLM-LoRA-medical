
CUDA_VISIBLE_DEVICES=0 python finetune_chatGLM.py \
    --base_model='../chatglm6b-dddd'\
    --genrate_data_dir='../genrate_data_dir'\
    --output_dir='output'