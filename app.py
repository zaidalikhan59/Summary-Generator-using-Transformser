# fast api
from fastapi import FastAPI , Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration , T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialization our fast api
app = FastAPI(title ="Text Summmarizer App" , description = "Text Summarization using T5" , version = "1.0")

# model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory = ".")

# input schema for dialogue
class DialogueInput(BaseModel):
    dialogue: str

# data cleaning 
def cleaned_data(text):
    text = re.sub(r"\r\n" , " " , text)#lines
    text = re.sub(r"\s+" , " " , text)#spaces
    text = re.sub(r"<.*?>" , " " , text)#html tags
    text = text.strip().lower()
    return text

# define summarization
def summarization(dialogue : str) -> str:
    dialogue = cleaned_data(dialogue)

    #tokens
    inputs = tokenizer(
        dialogue,
        max_length = 512,
        truncation = True,
        return_tensors = "pt"
    ).to(device)
    # generate summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4, # four outputs and give us best output
        early_stopping = True
    )
    #token ids => text (decoding)
    summary = tokenizer.decode(targets[0] , skip_special_tokes = True)# eg = EOS , SEP etc
    return summary

# Api Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarization(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/" , response_class = HTMLResponse)
async def home(request: Request):
     return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )