from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import os
import requests

class OpenAIImageInput(BaseModel):
    image_url: str = Field(..., description="The URL of the image to analyze.")
    prompt: Optional[str] = Field(default="Analyze this X-ray for anomalies", description="The analysis prompt")

class CircleTheAnomalyInput(BaseModel):
    image_url: str = Field(..., description="The URL of the X-ray image to analyze.")
    locations: list[dict] = Field(..., description="A list of dictionaries containing the coordinates of the anomalies to circle. Each dictionary should have 'x1', 'y1', 'x2', and 'y2' keys representing the top-left and bottom-right corners of the bounding box to circle.")

class OpenAIImageTool(BaseTool):
    name: str = "openai_image_tool"
    description: str = (
        "Analyzes X-ray images. "
        "Input: image_url (string) - Path to the image file, provided in the task description. "
    )
    args_schema: Type[BaseModel] = OpenAIImageInput

    def _run(self, image_url: str, prompt: Optional[str]) -> str:
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
                    {"type": "text", "text": "Describe this X-ray"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]

        url = "http://localhost:8081/v1/chat/completions"

        payload = {
            "model": "dale",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        # ✅ CORRETO: Envia o payload
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n=== RESPOSTA ===")
            return result["choices"][0]["message"]["content"]
        else:
            print("\n=== ERRO ===")
            print(response.text)  # ✅ Mostra o erro completo


class CircleTheAnomalyTool(BaseTool):
    name: str = "circle_the_anomaly"
    description: str = "A tool to circle anomalies in X-ray images, such as lung problems, spots, and secretions. If no anomalies are found, it should indicate that as well."
    args_schema: Type[BaseModel] = CircleTheAnomalyInput

    def _run(self, input: CircleTheAnomalyInput) -> str:
        """This is a placeholder implementation. In a real-world scenario, this method would contain the logic to analyze the X-ray image and circle the anomalies. For demonstration purposes, it simply returns a message indicating that the tool has been executed."""

        img = Image.open(input.image_url)

        # Cria o objeto de desenho
        draw = ImageDraw.Draw(img)  # ✅ Use ImageDraw.Draw()

        # Desenha um círculo vermelho
        draw.ellipse((input.locations[0]['x1'], input.locations[0]['y1'], input.locations[0]['x2'], input.locations[0]['y2']), outline='red', width=5)

        # Salva a imagem modificada
        output_path = "/home/lucas.abner/Documentos/code/med-crew/raio-x_annotated.jpeg"
        img.save(output_path)

        return f"Imagem salva em: {output_path}"
