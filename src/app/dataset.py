from datasets import Dataset, Audio, DatasetDict

audio_dataset = Dataset.from_dict({
    "audio": [
        "src/resources/srEdstva/Запись-_14_.mp3",
        "src/resources/srEdstva/Запись-_15_.mp3",
        "src/resources/srEdstva/Запись-_16_.mp3",
        "src/resources/srEdstva/Запись-_17_.mp3",
        "src/resources/srEdstva/Неправильно-1.mp3",
        "src/resources/srEdstva/Неправильно-2.wav",
        "src/resources/srEdstva/Неправильно-3.mp3",
        "src/resources/srEdstva/Неправильно-4.mp3",
        "src/resources/srEdstva/Неправильно.mp3",
        "src/resources/srEdstva/Правильно-1.mp3",
        "src/resources/srEdstva/Правильно-2.wav",
        "src/resources/srEdstva/Правильно-3.mp3",
        "src/resources/srEdstva/Правильно-4.mp3",
        "src/resources/srEdstva/Правильно-5.mp3",
        "src/resources/srEdstva/Правильно.mp3"
        ],
    "transalation": [
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "СрЕдства",
        "Средства",
        "Средства",
        "Средства",
        "Средства",
        "Средства",
        "Средства"
        ]
    }).cast_column("audio", Audio())
audio_dataset[0]["audio"]

audio_dataset_test = Dataset.from_dict({
    "audio": [
        "src/resources/srEdstva/Запись-_14_.mp3",
        "src/resources/srEdstva/Неправильно-1.mp3",
        "src/resources/srEdstva/Правильно-1.mp3",
        "src/resources/srEdstva/Правильно.mp3"
        ],
    "transalation": [
        "СрЕдства",
        "СрЕдства",
        "Средства",
        "Средства"
        ]
    }).cast_column("audio", Audio())
audio_dataset[0]["audio"]

common_voice = DatasetDict()

common_voice["train"] = audio_dataset
common_voice["test"] = audio_dataset_test

print(common_voice)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
