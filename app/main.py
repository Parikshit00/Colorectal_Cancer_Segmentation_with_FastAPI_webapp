from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import utils
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.post("/")
async def home_predict(request: Request, file: UploadFile = File(...)):
    result = None
    error = None
    try:
        result = utils.get_result(image_file=file)
    except Exception as ex:
        error = ex
    return templates.TemplateResponse("index.html", {"request": request, "result": result , "error": error})
