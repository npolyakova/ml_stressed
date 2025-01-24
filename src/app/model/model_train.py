# if windows run win_helper first

import evaluate
from datasets import DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from src.app.model.DataCollator import DataCollatorSpeechSeq2SeqWithPadding
from src.app.model.dataset import create_dataset, get_data

# create datasets
audio_train, translation_train = get_data('train_data.csv')
audio_training_dataset = create_dataset(audio_train, translation_train)
audio_training_dataset.push_to_hub("TigrulyaCat/stressed_syllables")

audio_test, translation_test = get_data('test_data.csv')
audio_validation_dataset = create_dataset(audio_test, translation_test)

common_voice = DatasetDict()
common_voice["train"] = audio_training_dataset
common_voice["test"] = audio_validation_dataset

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="russian", task="transcribe")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["translation"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, num_proc=1)

# set metrics
metric = evaluate.load("wer")

# train

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="russian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model.generation_config.language = "russian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=5,
    max_steps=20,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    ),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

#tokenizer.push_to_hub("TigrulyaCat/whisper-small-hi")
trainer.push_to_hub()