import json
import re
import os
import ast

from gradio import Interface, Image, File
from pathlib import Path 
from medgemma_crew.main import run

# Importe a sua classe aqui!
from medgemma_crew.tools.docx_transforming import FerramentaEscreverDocx

def process_ray_x_image(image_path: str) -> str:
    """Função para processar a imagem de raio-x e retornar o caminho do relatório gerado."""
    
    response_crew = run(image_path)
    # response_raw = response_crew.raw
    # response_json = response_crew.json_dict

    # print("Resposta completa do Crew:")
    # print(response_raw)  # ✅ Mostra a resposta completa do Crew para análise

    # print("\nResposta JSON do Crew:")
    # print(json.dumps(response_json, indent=2))  # ✅ Mostra a resposta JSON para verificar a estrutura e os dados retornados
    
    result_str = str(response_crew)
    # print("\n=== RESULTADO EXTRAÍDO ===")
    # print(result_str)  # ✅ Mostra o resultado extraído para verificar o formato
    
    # Extrai o JSON gerado pela write_report_task
    match = re.search(r'\{.*\}', result_str, re.DOTALL)
    # print("\n=== JSON EXTRAÍDO ===")
    # print(match.group(0) if match else "Nenhum JSON encontrado")  # ✅ Mostra o JSON extraído ou mensagem de erro
    
    if match:
        dict_str = match.group(0)
        try:
            dados = ast.literal_eval(dict_str)  # Converte a string do dicionário para um objeto Python
            print("\n=== DADOS DECODIFICADOS ===")
            print(json.dumps(dados, indent=2))  # ✅ Mostra os dados decodificados para verificar o conteúdo
            
            # 1. Montamos o conteúdo em Markdown usando os dados do JSON
            conteudo_markdown = f"""# Radiology Report

## Summary
{dados.get('summary', '')}

## Anomaly Description
{dados.get('anomaly_description', '')}

## Diagnosis
### Main
{dados.get('diagnosis', {}).get('main', '')}

### Differential
{', '.join(dados.get('diagnosis', {}).get('differential', []))}

## Recommendations
{dados.get('recommendations', '')}

## Image Paths
![Heatmap]({dados.get('heatmap_image_path', '')})

![Annotated Image]({dados.get('annotated_image_path', '')})

## Visual Comparison
{dados.get('visual_comparison', '')}
"""
            
            # 2. Definimos onde salvar o arquivo (caminho absoluto recomendado)
            # Salva na mesma pasta do script rodando
            caminho_docx = os.path.abspath("relatorio_radiologico_final.docx")
            
            # 3. INSTANCIAMOS A SUA CLASSE E CHAMAMOS O _run()
            ferramenta_docx = FerramentaEscreverDocx()
            resultado_ferramenta = ferramenta_docx._run(
                caminho_ficheiro=caminho_docx, 
                conteudo=conteudo_markdown
            )
            print("\n=== RESULTADO DA FERRAMENTA ===")
            print(resultado_ferramenta)

            # --- INÍCIO DA CORREÇÃO DE VALIDAÇÃO ---
            # Verifica se o resultado é um dicionário e se teve sucesso
            print(f"Caminho do arquivo PDF: {resultado_ferramenta['pdf_path']}")

            # Apagando imagens temporárias...
            caminho_output = Path(__file__).parent.parent
            caminho_imgs = os.path.join(caminho_output, "medgemma_crew/outputs")
            print(f"Caminho da pasta de imagens temporárias: {caminho_imgs}")
            if os.path.exists(caminho_imgs):
                items =os.listdir(caminho_imgs)
                for item in items:
                    item_path = os.path.join(caminho_imgs, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                print(f"Imagens temporárias removidas da pasta: {caminho_imgs}")
            
            return resultado_ferramenta['pdf_path']

        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            return None
    else:
        print("Nenhum JSON válido foi encontrado na resposta.")
        return None


# Configuração do Gradio
interface = Interface(
    fn=process_ray_x_image, 
    inputs=[Image(type="filepath")], 
    outputs=[File(label="Relatório Médico em pdf", type="filepath")], 
    title="Medgemma Crew", 
    flagging_dir=Path(__file__).parent / "logs"
)

if __name__ == "__main__":
    interface.launch()