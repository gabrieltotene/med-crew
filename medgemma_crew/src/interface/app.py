from gradio import Interface,Image, File
from medgemma_crew.main import run



interface = Interface(fn=run, inputs=[Image(type="filepath")], outputs=[File(type="filepath")], title="Medgemma Crew")

interface.launch()