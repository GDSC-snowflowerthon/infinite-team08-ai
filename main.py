from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

class ImagePromptRequest(BaseModel):
    prompt: str

client = OpenAI(
    api_key="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"
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
