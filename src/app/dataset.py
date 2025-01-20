from datasets import Dataset, Audio, DatasetDict

audio_dataset = Dataset.from_dict({
    "audio": [
        "Запись-_14_.mp3",
        "Запись-_15_.mp3",
        "Запись-_16_.mp3",
        "Запись-_17_.mp3",
        "Неправильно-1.mp3",
        "Неправильно-2.wav",
        "Неправильно-3.mp3",
        "Неправильно-4.mp3",
        "Неправильно.mp3",
        "Правильно-1.mp3",
        "Правильно-2.wav",
        "Правильно-3.mp3",
        "Правильно-4.mp3",
        "Правильно-5.mp3",
        "Правильно.mp3"
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
        "Запись-_14_.mp3",
        "Неправильно-1.mp3",
        "Правильно-1.mp3",
        "Правильно.mp3"
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
