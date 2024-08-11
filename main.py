from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio
from fastapi.middleware.cors import CORSMiddleware
import os
from concurrent.futures import ThreadPoolExecutor
from SentimentAnalysis import get_sentiment
from pydantic import BaseModel
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)

@app.post("/receiveAudio")
async def receiveAudio(file: UploadFile = File(...)):
    file_path = f"audios/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())


    future = executor.submit(transcribe, file_path)
    result = future.result()

    os.remove(file_path)

    return {"transcription": result}

@app.post("/receiveAudio/transcribe")
async def receiveAudio(file: UploadFile = File(...)):
    file_path = f"audios/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())


    future = executor.submit(transcribe, file_path)
    result = future.result()
    Sentiment = get_sentiment(result[0])
    #Sentiment = 0

    os.remove(file_path)

    return {"transcription": result, "sentiment": Sentiment}

def transcribe(file_path):
    model_size = "small.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    file = decode_audio(file_path)
    segments, info = batched_model.transcribe(file, beam_size=5, language="en")
    result = []
    for segment in segments:
        result.append(segment.text)
    return result



@app.get("/trans")
async def trans():
    model_size = "small.en"
    model = WhisperModel(model_size)
    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe("No Look C4.mp3")
    result = []
    for segment in segments:
        result.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return result


class TextModel(BaseModel):
    text: str

@app.post("/sentiment")
async def receiveAudio(text:TextModel):
    Sentiment = get_sentiment(text.text)
    return Sentiment

@app.get("/test")
async def test():
    return "Hello World"





