from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

class ImagePromptRequest(BaseModel):
    prompt: str

class TranslationRequest(BaseModel):
    text: str

client = OpenAI(
    api_key="sk-"
)

@app.post("/process_image/")
async def process_image(request: ImageRequest):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": request.image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

@app.post("/generate_image/")
async def generate_image(request: ImagePromptRequest):
    response = client.images.generate(
        model="dall-e-3",
        prompt=request.prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return {"image_url": image_url}

@app.post("/translate_to_korean/")
async def translate_to_korean(request: TranslationRequest):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a translation assistant. Translate from English to Korean."},
            {"role": "user", "content": request.text}
        ]
    )

    translated_text = response.choices[0].message.content

    return {"translated_text": translated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
