import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainer_state_dir', default='../output/checkpoint-5200/trainer_state.json', 
                    type=str,help='trainer_state_dir')
parser.add_argument('--save',default=False,
                    type=bool,help='save')

if __name__=="__main__":
    args = parser.parse_args()
    trainer_state_dir = args.trainer_state_dir
    save = args.save
    with open(trainer_state_dir, 'r') as f:
        trainer_state = json.load(f)

    df=pd.DataFrame.from_dict(trainer_state['log_history'])
    df_merged = df.groupby('epoch').first().reset_index()   
    if save:
        df_merged.to_csv('output.csv', index=False)

    sns.lineplot(data=df_merged, x='epoch', y='eval_loss', label='Eval Loss')

    sns.lineplot(data=df_merged, x='epoch', y='loss', label='Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    if save:
        plt.savefig('Loss_Curve.png')
    plt.show()