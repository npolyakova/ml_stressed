from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, \
    WhisperTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

feature_extractor = WhisperFeatureExtractor.from_pretrained("src/app/model/whisper-small-hi/checkpoint-10")
tokenizer = WhisperTokenizer.from_pretrained("app/model/whisper-small-hi", language="russian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("app/model/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("app/model/whisper-small-hi")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    torch_dtype=torch_dtype,
)

result = pipe("resources/srEdstva/validation/test_1.mp3", generate_kwargs={"language": "russian"})
print(result["text"])