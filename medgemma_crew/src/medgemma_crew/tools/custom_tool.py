from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import os
import requests
from pathlib import Path

class OpenAIImageInput(BaseModel):
    image_url: str = Field(..., description="The URL of the X-ray image to analyze.")
    prompt: str = Field(default="Analyze this X-ray for anomalies", description="The analysis prompt")

# class CircleTheAnomalyInput(BaseModel):
#     image_url: str = Field(..., description="The URL of the X-ray image to analyze.")
#     locations: list[dict] = Field(..., description="A list of dictionaries containing the coordinates of the anomalies to circle. Each dictionary should have 'x1', 'y1', 'x2', and 'y2' keys representing the top-left and bottom-right corners of the bounding box to circle.")

class OpenAIImageTool(BaseTool):
    name: str = "xray_image_tool"
    description: str = (
        "Analyzes X-ray images. "
        "Input: image_url (string) - Path to the image file, provided in the task description. "
    )
    args_schema: Type[BaseModel] = OpenAIImageInput

    def _run(self, image_url: str, prompt: str) -> str:
        """Analyzes an image and returns description."""
        print(f"Received image URL: {image_url}")
        def encode_base64(image_url: str) -> str:
            with open(image_url, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        print('Encoding image...')
        img_base64 = encode_base64(image_url)
        print(f'Image encoded: {len(img_base64)} characters')

        # Mensagens no formato correto
        messages = [
            {
                "role": "system",
                "content": "You are an expert radiologist."  # ✅ Simplificado
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]

        url = "http://localhost:11434/v1/chat/completions"

        payload = {
            "model": "thiagomoraes/medgemma-1.5-4b-it:Q8_0",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.0
        }

        # ✅ CORRETO: Envia o payload
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n=== RESPOSTA ===")

            content = result["choices"][0]["message"]["content"]
            return content

        else:
            print("\n=== ERRO ===")
            print(response.text)  # ✅ Mostra o erro completo


# class CircleTheAnomalyTool(BaseTool):
#     name: str = "circle_the_anomaly"
#     description: str = (
#         "Draws red circles on X-ray images using normalized 0-1000 bounding boxes."
#         "Input: image_url (string) - Path to the image file, provided in the task description. "
#     )
#     args_schema: Type[BaseModel] = CircleTheAnomalyInput

#     def _run(self, image_url: str, locations: list[dict]) -> str:

#         if image_url.startswith("file://"):
#             image_url = image_url.replace("file://", "")

#         if not image_url or not os.path.exists(image_url):
#             return f"Error: Image file not found at {image_url}"

#         img = Image.open(image_url).convert("RGB")
#         width, height = img.size

#         draw = ImageDraw.Draw(img)

#         if not locations or len(locations) == 0:
#             return "No locations provided."

#         for loc in locations:

#             # 🔒 Validação defensiva
#             if not all(k in loc for k in ["y1", "x1", "y2", "x2"]):
#                 continue

#             ymin = int(loc["y1"])
#             xmin = int(loc["x1"])
#             ymax = int(loc["y2"])
#             xmax = int(loc["x2"])

#             # 🔒 Clamp de segurança
#             ymin = max(0, min(1000, ymin))
#             xmin = max(0, min(1000, xmin))
#             ymax = max(0, min(1000, ymax))
#             xmax = max(0, min(1000, xmax))

#             if ymin >= ymax or xmin >= xmax:
#                 continue

#             # 📐 Conversão para pixel real
#             left = int(xmin * width / 1000)
#             right = int(xmax * width / 1000)
#             top = int(ymin * height / 1000)
#             bottom = int(ymax * height / 1000)

#             # 🎯 Desenha elipse
#             draw.ellipse([left, top, right, bottom], outline="red", width=5)

#         # 📁 Gera nome único
#         import uuid
#         path_caminho = Path(__file__).parent
#         output_path = f"{path_caminho}/raio-x_annotated_{uuid.uuid4().hex[:8]}.jpeg"
        

#         img.save(output_path)

#         return output_path
