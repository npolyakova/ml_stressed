import csv
from pathlib import Path

from datasets import Dataset, Audio

def create_dataset(audio: [str], translation: [str]):
    return Dataset.from_dict({
        "audio": audio,
        "translation": translation
    }).cast_column("audio", Audio())

def get_data(file_name: str):
    with open(file_name, newline='', encoding="utf-8") as csvfile:
        audio = []
        translation = []
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            audio.append(row[0])
            translation.append(row[1])
        return audio, translation
