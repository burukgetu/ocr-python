from fastapi import FastAPI, UploadFile, File
import easyocr
import cv2
import shutil

app = FastAPI()

reader = easyocr.Reader(['en'], gpu=False)

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Failed to read image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(gray)
        extracted_text = "\n".join([detection[1] for detection in result])

        return {"text": extracted_text}
    except Exception as e:
        return {"error": str(e)}
