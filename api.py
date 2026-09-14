from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from load import predict_image


app = FastAPI(
    title="MNIST CNN API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MNIST CNN API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    temp_path = None

    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Empty image"
            )

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        result, prob = predict_image(temp_path)

        # prediction در پروژه فعلی int است
        prediction = int(result)

        # confidence از Tensor probability گرفته می‌شود
        confidence = float(prob.max())

        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
