#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from medgemma_crew.crew import MedgemmaCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    img = "/home/lucas.abner/Documentos/code/med-crew/raio_4.png"
    img2 = "C:\\Users\\asus\\Documents\\IA\\med-crew\\raio_4.png"

    inputs = {
        'image_url': img
    }
    
    try:
        MedgemmaCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")