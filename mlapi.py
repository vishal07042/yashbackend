from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this import

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yasgfrdeepfake.vercel.app","http://localhost:5173"], 
     # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... rest of your API code

# Load the pre-trained model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

class ImagePrediction(BaseModel):
    isReal: bool

def preprocess_image(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class == 0  # Return True if predicted as real, False otherwise

@app.post('/')
async def predictImage(image: UploadFile):
    is_real = preprocess_image(await image.read())
    return ImagePrediction(isReal=is_real)
