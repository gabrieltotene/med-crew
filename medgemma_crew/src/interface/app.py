from gradio import Interface,Image
from medgemma_crew.main import run



interface = Interface(fn=run, inputs=[Image(type="filepath")], outputs="text", title="Medgemma Crew")

interface.launch()