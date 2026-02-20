from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel
from .tools.custom_tool import OpenAIImageTool
from .tools.xray_anomaly_locator import XRayAnomalyLocatorTool
from .tools.docx_transforming import FerramentaEscreverDocx


# Modelos de saída JSON
class XRayAnalysisOutput(BaseModel):
    anomaly_found: bool
    anomalies: list[dict]
    normal_structures: str
    image_quality: str
    technical_description: str


class RadiologyReportOutput(BaseModel):
    technique: str
    findings: str
    impression: dict
    recommendations: str
    full_report: str


class AnnotatedImageOutput(BaseModel):
    annotated_image_path: str
    pathologies_detected: list[dict]
    anomaly_found: bool

class FinalReportOutput(BaseModel):
    summary: str
    anomaly_description: str
    diagnosis: dict
    recommendations: str
    original_image_path: str
    annotated_image_path: str
    visual_comparison: str


@CrewBase
class MedgemmaCrew():
    """MedgemmaCrew - Análise de Raio-X com 3 agentes"""

    llm = LLM(
        model="ollama/thiagomoraes/medgemma-1.5-4b-it:Q8_0",
        api_key="",
        base_url="http://localhost:11434",
        supports_function_calling=True,
        temperature=0.0,
    )

    llm_tool = LLM(
        model="ollama/qwen2.5:7b",
        api_key="",
        base_url="http://localhost:11434"
    )

    agents: List[BaseAgent]
    tasks: List[Task]

    # ==================== AGENTES ====================
    
    @agent
    def xray_analyzer(self) -> Agent:
        """Agente 1: Analisa o raio-x com MedGemma"""
        return Agent(
            config=self.agents_config['xray_analyzer'],
            verbose=True,
            llm=self.llm_tool,
            max_execution_time=600
        )

    @agent
    def report_writer(self) -> Agent:
        """Agente 2: Redige o relatório médico"""
        return Agent(
            config=self.agents_config['report_writer'],
            verbose=True,
            llm=self.llm,
            max_execution_time=600
        )

    @agent
    def image_annotator(self) -> Agent:
        """Agente 3: Circula anomalias e salva imagem"""
        return Agent(
            config=self.agents_config['image_annotator'],
            verbose=True,
            llm=self.llm_tool,
            max_execution_time=900
        )

    @agent
    def redator(self) -> Agent:
        return Agent(
            config = self.agents_config['redator'],
            verbose=True,
            llm=self.llm_tool,
            max_execution_time=600
        )
    # ==================== TASKS ====================
    
    @task
    def xray_analysis_task(self) -> Task:
        """Task 1: Análise do raio-x com MedGemma"""
        return Task(
            config=self.tasks_config['xray_analysis_task'],
            tools=[OpenAIImageTool()],
            output_json=XRayAnalysisOutput,
        )

    @task
    def report_writing_task(self) -> Task:
        """Task 2: Redação do relatório médico"""
        return Task(
            config=self.tasks_config['report_writing_task'],
            context=[self.xray_analysis_task()],
            output_json=RadiologyReportOutput,
            async_execution=True
        )

    @task
    def annotate_image_task(self) -> Task:
        """Task 3: Circular anomalia e salvar imagem"""
        return Task(
            config=self.tasks_config['annotate_image_task'],
            tools=[XRayAnomalyLocatorTool()],
            output_json=AnnotatedImageOutput,
            async_execution=True
        )
    
    @task
    def write_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_report_task'],
            context=[self.xray_analysis_task(), self.annotate_image_task()],
            output_json=FinalReportOutput,
            markdown_output=True
        )
    
    @task
    def show_report_docx(self) -> Task:
        return Task(
            config=self.tasks_config['show_report_docx'],
            tools=[FerramentaEscreverDocx()],
            context=[self.write_report_task()],
        )

    # ==================== CREW ====================
    
    @crew
    def crew(self) -> Crew:
        """Cria o crew com os 3 agentes em sequência"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )
