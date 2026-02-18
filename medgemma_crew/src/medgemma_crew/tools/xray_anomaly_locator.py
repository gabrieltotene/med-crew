from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, ClassVar
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchxrayvision as xrv
import os
import uuid
from pathlib import Path
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class XRayAnomalyLocatorInput(BaseModel):
    image_url: str = Field(..., description="Path to the X-ray image file")
    pathology: str = Field(default="auto", description="Pathology to locate. Use 'auto' to detect highest probability.")


class XRayAnomalyLocatorTool(BaseTool):
    name: str = "xray_anomaly_locator"
    description: str = (
        "Locates and circles anomalies in chest X-ray images. "
        "Uses AI to detect pathology regions and draws circles on the ORIGINAL image. "
        "Input: image_url (required), pathology (optional, default 'auto'), threshold (optional, default 0.6)."
    )
    args_schema: Type[BaseModel] = XRayAnomalyLocatorInput
    
    model: Optional[object] = None
    pathologies: Optional[list] = None
    global_model: ClassVar[Optional[object]] = None
    global_pathologies: ClassVar[Optional[list]] = None

    def _load_model(self):
        if XRayAnomalyLocatorTool.global_model is None:
            print("[XRayAnomalyLocatorTool] Carregando TorchXRayVision global...")
            XRayAnomalyLocatorTool.global_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            XRayAnomalyLocatorTool.global_model.eval()
            XRayAnomalyLocatorTool.global_pathologies = list(xrv.datasets.default_pathologies)
        self.model = XRayAnomalyLocatorTool.global_model
        self.pathologies = XRayAnomalyLocatorTool.global_pathologies

    def _run(self, image_url: str, pathology: str = "auto") -> dict:
        print(f"[XRayAnomalyLocatorTool] Iniciando _run para {image_url} com pathology={pathology}")
        self._load_model()
        
        if image_url.startswith("file://"):
            image_url = image_url.replace("file://", "")
        
        if not os.path.exists(image_url):
            print(f"[XRayAnomalyLocatorTool] ERRO: Imagem não encontrada: {image_url}")
            return {"error": f"Image not found: {image_url}"}
        
        try:
            # Carrega imagem original
            original_img = Image.open(image_url).convert("RGB")
            orig_width, orig_height = original_img.size
            
            # Prepara para o modelo
            img_gray = Image.open(image_url).convert("L")
            img_resized = img_gray.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img_resized)
            img_normalized = xrv.datasets.normalize(img_array, 255)
            img_normalized = img_normalized[None, ...]
            
            img_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0)
            img_tensor.requires_grad = True
            
            # Forward pass
            output = self.model(img_tensor)
            probs = torch.sigmoid(output).detach().numpy()[0]
            
            # Encontra patologias detectadas
            detections = []
            for p, prob in zip(self.pathologies, probs):
                if prob >= 0.6:
                    detections.append({"pathology": p, "probability": round(float(prob), 3)})
            detections.sort(key=lambda x: x["probability"], reverse=True)
            
            if not detections:
                print("[XRayAnomalyLocatorTool] Nenhuma patologia significativa detectada.")
                return {"anomaly_found": False, "message": "No significant pathologies detected."}
            
            # Escolhe patologia
            if pathology == "auto" or pathology not in self.pathologies:
                target_pathology = detections[0]["pathology"]
            else:
                target_pathology = pathology
            
            pathology_idx = self.pathologies.index(target_pathology)
            
            # Backward pass - gradientes simples
            self.model.zero_grad()
            output[0, pathology_idx].backward()
            
            # Pega gradientes da entrada
            gradients = img_tensor.grad[0, 0].numpy()
            
            # Usa gradientes positivos (regiões que aumentam a predição)
            heatmap = np.maximum(gradients, 0)
            
            # Suaviza o heatmap
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=5)
            
            # Normaliza
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Salva visualização do heatmap (seguro em ambientes headless)
            heatmap_path = None
            try:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img_normalized[0], cmap="gray")
                axes[0].set_title("X-Ray Original")
                axes[0].axis("off")
                axes[1].imshow(img_normalized[0], cmap="gray")
                axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
                axes[1].set_title(f"Heatmap: {target_pathology}")
                axes[1].axis("off")
                plt.tight_layout()
                heatmap_path = str(Path(__file__).parent / f"heatmap_{uuid.uuid4().hex[:8]}.png")
                fig.savefig(heatmap_path, bbox_inches="tight")
                plt.close(fig)
                print(f"[XRayAnomalyLocatorTool] Heatmap salvo em {heatmap_path}")
            except Exception as ex:
                print(f"[XRayAnomalyLocatorTool] Falha ao salvar heatmap: {ex}")
                heatmap_path = None
            
            # Threshold adaptativo baseado no percentil 85
            threshold = np.percentile(heatmap, 85)
            binary_mask = heatmap > threshold
            
            # Encontra componentes conectados
            labeled_array, num_features = ndimage.label(binary_mask)
            
            regions = []
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    intensity = heatmap[region_mask].mean()
                    area = region_mask.sum()
                    
                    # Filtra regiões muito pequenas
                    if area > 50:
                        regions.append({
                            "bbox_224": [int(x1), int(y1), int(x2), int(y2)],
                            "intensity": float(intensity),
                            "area": int(area)
                        })
            
            regions.sort(key=lambda x: x["intensity"], reverse=True)
            top_regions = regions[:1]  # Apenas a região principal
            
            # Desenha na imagem original
            draw = ImageDraw.Draw(original_img)
            
            scale_x = orig_width / 224
            scale_y = orig_height / 224
            
            circled_regions = []
            
            # Tamanhos
            min_size = int(min(orig_width, orig_height) * 0.15)
            max_size = int(min(orig_width, orig_height) * 0.35)
            
            for region in top_regions:
                x1, y1, x2, y2 = region["bbox_224"]
                
                # Escala para original
                orig_x1 = int(x1 * scale_x)
                orig_y1 = int(y1 * scale_y)
                orig_x2 = int(x2 * scale_x)
                orig_y2 = int(y2 * scale_y)
                
                # Centro
                center_x = (orig_x1 + orig_x2) // 2
                center_y = (orig_y1 + orig_y2) // 2
                
                # Tamanho com limites
                width = max(orig_x2 - orig_x1, min_size)
                height = max(orig_y2 - orig_y1, min_size)
                width = min(width, max_size)
                height = min(height, max_size)
                
                # Margem 25%
                width = int(width * 1.25)
                height = int(height * 1.25)
                width = min(width, max_size)
                height = min(height, max_size)
                
                # Coordenadas finais
                final_x1 = max(0, center_x - width // 2)
                final_y1 = max(0, center_y - height // 2)
                final_x2 = min(orig_width, center_x + width // 2)
                final_y2 = min(orig_height, center_y + height // 2)
                
                draw.ellipse([final_x1, final_y1, final_x2, final_y2], outline="red", width=5)
                
                circled_regions.append({
                    "coordinates": [final_x1, final_y1, final_x2, final_y2]
                })
            
            # Salva
            output_dir = Path(__file__).parent
            output_path = str(output_dir / f"xray_circled_{uuid.uuid4().hex[:8]}.png")
            original_img.save(output_path)
            print(f"[XRayAnomalyLocatorTool] Imagem anotada salva em {output_path}")
            result = {
                "anomaly_found": True,
                "output_image": output_path,
                "heatmap_image": heatmap_path,
                "target_pathology": target_pathology,
                "pathology_probability": float(probs[pathology_idx]),
                "circled_regions": circled_regions,
                "all_detections": detections
            }
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}