import os
from typing import List
from SoccerNet.Downloader import SoccerNetDownloader

def download_raw_games_from(soccernet_downloader: SoccerNetDownloader, games: List[str]):
    for game in games:
        soccernet_downloader.downloadGame(
            game=game,
            files=['Frames-v3.zip', 'Labels-v3.json']
        )

        os.system(f'yes | unzip "data/raw/{game}/Frames-v3.zip" -d "/ata/raw/{game}/frames"')


def download_calibration_task_from(soccernet_downloader: SoccerNetDownloader, split=["train", "valid", "test", "challenge"]):
    soccernet_downloader.downloadDataTask(task="calibration", split=split)
    os.system('yes | unzip data/calibration/train.zip -d data/calibration/')
    os.system('yes | unzip data/calibration/test.zip -d data/calibration/')
    os.system('yes | unzip data/calibration/valid.zip -d data/calibration/')
    os.system('yes | unzip data/calibration/challenge.zip -d data/calibration/challenge')

if __name__ == '__main__':
    # download_raw_games_from(
    #     SoccerNetDownloader(LocalDirectory="data/raw"),
    #     games=[
    #         'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley',
    #     ]
    # )

    download_calibration_task_from(
        SoccerNetDownloader(LocalDirectory="data"),
        split=['train', 'test', 'valid']
    )
        