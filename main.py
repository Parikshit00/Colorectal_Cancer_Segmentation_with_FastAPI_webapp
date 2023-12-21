from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})
