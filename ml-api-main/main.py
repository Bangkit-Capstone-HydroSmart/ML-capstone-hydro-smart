import os
import uvicorn
import traceback
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, Response

# Initialize Model
model = tf.keras.models.load_model('./plant_rec_model.h5')

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# Endpoint for text input
class RequestText(BaseModel):
    text:str

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        text = req.text
        print("Uploaded text:", text)
        
        # Step 1: (Optional) Do your text preprocessing
        def preprocess_text(text):
            tokenized_text = text.split()
            return tokenized_text
        tokenized_text = preprocess_text(text)

        # Step 2: Prepare your data to your model
        prepared_data = tokenized_text

        # Step 3: Predict the data
        result = model.predict([prepared_data])
        
        # Step 4: Change the result your determined API output
        def format_output(result):
        # Mengembalikan hasil prediksi sebagai string
            return str(result)
        api_output = format_output(result)

        return {"api_output": api_output}
        
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)