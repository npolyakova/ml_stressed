import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("TigrulyaCat/whisper-small-hi")
model = AutoModelForSpeechSeq2Seq.from_pretrained("TigrulyaCat/whisper-small-hi")

pipe = pipeline("automatic-speech-recognition", model="TigrulyaCat/whisper-small-hi")

result = pipe("model/resources/aeropOrty/validation/test_4.mp3", generate_kwargs={"language": "russian"})
print(result["text"])
