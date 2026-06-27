import os
from typing import List
from dotenv import load_dotenv, dotenv_values
from SoccerNet.Downloader import SoccerNetDownloader

"""
def download_raw_games_from(soccernet_downloader: SoccerNetDownloader, games: List[str]):
    for game in games:
        soccernet_downloader.downloadGame(
            game=game,
            files=['Frames-v3.zip', 'Labels-v3.json']
        )

        os.system(f'yes | unzip "data/raw/{game}/Frames-v3.zip" -d "/data/raw/{game}/frames"')
"""


def download_task_from(
    data_folder: str, task: str,
    split=["train", "valid", "test", "challenge"], **kwargs
):
    os.system(f'ls -l {data_folder}/{task}')

    soccernet_downloader = SoccerNetDownloader(LocalDirectory=data_folder)
    print("- Download dataset from SoccerNet, task =", task)

    soccernet_downloader.downloadDataTask(task=task, split=split, **kwargs)
    for split_type in split:
        print(f'Executing: yes | unzip -q {data_folder}/{task}/{split_type}.zip -d {data_folder}/{task}/')
        os.system(f'yes | unzip -q {data_folder}/{task}/{split_type}.zip -d {data_folder}/{task}/')

if __name__ == '__main__':
    __ENV = dotenv_values(".env")
    __DATA_FOLDER = __ENV['DATA_FOLDER']
    
    # download_task_from(
    #     data_folder=__DATA_FOLDER,
    #     task='SpiideoSynLoc',
    #     split=['train', 'test', 'valid', 'challenge'],
    #     version='fullhd'
    # )

    download_task_from(
        data_folder=__DATA_FOLDER,
        task='calibration',
        split=['train', 'test', 'valid'],
        version='fullhd'
    )
