import argparse
from glob import glob
from itertools import chain
import pandas as pd 
import os
from tqdm import tqdm
from pathlib import Path
import shutil
from typing import Optional
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--genrate_data_dir', default='./genrate_data_dir', 
                    type=str,help='genrate_data_dir')
parser.add_argument('--target_dir_list', 
                    default=['./chinese_dataset/其他中文问题补充/',
                   './chinese_dataset/翻译后的中文数据/',
                   './chinese_dataset/医学领域数据/',
                #    './chinese_dataset/chatglm问题数据补充/',
                #    'chinese_dataset/原始英文数据/'
                   ],
                    type=list,help='target_dir_list')

def read_json(x:str):
    try:
        data = pd.read_json(x)
        return data 
    except Exception as e:
        return pd.DataFrame()

if __name__ == "__main__":
    args = parser.parse_args()
    genrate_data_dir=Path(args.genrate_data_dir)
    target_dir_list=args.target_dir_list

    all_json_path = [glob(i+"*.json") for i in target_dir_list]
    all_json_path = list(chain(*all_json_path))
    print('总共有：{} 个json文件'.format(len(all_json_path)))

    alldata = pd.concat([read_json(i) for i in all_json_path])

    if genrate_data_dir.exists():
        shutil.rmtree(genrate_data_dir, ignore_errors=True)
    
    os.makedirs(genrate_data_dir, exist_ok=True)
    alldata = alldata.sample(frac=1).reset_index(drop=True)

    chunk_size = 666
    for index, start_id in tqdm(enumerate(range(0, alldata.shape[0], chunk_size))):
        temp_data = alldata.iloc[start_id:(start_id+chunk_size)]
        temp_data.to_csv(genrate_data_dir.joinpath(f"{index}.csv"), index=False)