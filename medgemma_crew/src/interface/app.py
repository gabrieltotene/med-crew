from gradio import Interface,Image, File
from pathlib import Path 
from medgemma_crew.main import run



interface = Interface(
    fn=run, 
    inputs=[Image(type="filepath")], 
    outputs=[File(type="filepath")], 
    title="Medgemma Crew", 
    flagging_dir=Path(__file__).parent / "logs",
    api_visibility="private",

)

interface.launch()

# Pra rodar
# cd medgemma_crew/src/
# python -m interface.app