import os
from typing import List
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader

def download_raw_games_from(soccernet_downloader: SoccerNetDownloader, games: List[str]):
    for game in games:
        soccernet_downloader.downloadGame(
            game=game,
            files=['Frames-v3.zip', 'Labels-v3.json']
        )

        os.system(f'yes | unzip "data/raw/{game}/Frames-v3.zip" -d "/ata/raw/{game}/frames"')


def download_task_from(
    soccernet_downloader: SoccerNetDownloader, task: str, 
    split=["train", "valid", "test", "challenge"], **kwargs
):
    soccernet_downloader.user = os.environ["SOCCERNET_SPIIDEO_USER"]
    soccernet_downloader.password = os.environ["SOCCERNET_SPIIDEO_PASSWORD"]

    soccernet_downloader.downloadDataTask(task=task, split=split, **kwargs)
    for split_type in split:
        os.system(f'yes | unzip data/{task}/{split_type}.zip -d data/{task}/')

if __name__ == '__main__':
    load_dotenv()

    download_task_from(
        SoccerNetDownloader(LocalDirectory="data"),
        task='SpiideoSynLoc',
        split=['train', 'test', 'valid', 'challenge'],
        version='fullhd'
    )
        
