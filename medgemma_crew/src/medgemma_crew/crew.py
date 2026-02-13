from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel
from .tools.custom_tool import OpenAIImageTool,CircleTheAnomalyTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class AnomalyJson(BaseModel):
    anomaly_found: bool
    anomalies: list[dict]  # List of anomalies with their descriptions and coordinates
    technical_description: str

class LocalizationJson(BaseModel):
    anomaly_found: bool # List of anatomical structures with their descriptions and coordinates
    anomalies: list[dict]

class RadiologyReport(BaseModel):
    pulmonary_problem: str
    probability: float
    detailed_report: str

@CrewBase
class MedgemmaCrew():
    """MedgemmaCrew crew"""

    llm = LLM(
        model="openai/dale",
        api_key=".",
        base_url="http://localhost:8081",
        supports_function_calling=True,
        temperature=0.0,
    )

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def image_interpreter(self) -> Agent:
        return Agent(
            config=self.agents_config['image_interpreter'], # type: ignore[index]
            verbose=True,
            # multimodal=True, # In case you want to use multimodal capabilities
            llm=self.llm
        )

    @agent
    def spatial_localizer(self) -> Agent:
        return Agent(
            config=self.agents_config['spatial_localizer'], # type: ignore[index]
            verbose=True,
            llm=self.llm
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            llm=self.llm
        )

    @agent
    def anomaly_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['anomaly_analyst'], # type: ignore[index]
            verbose=True,
            llm=self.llm,
            allow_delegation=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def image_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_analysis_task'], # type: ignore[index]
            tools=[OpenAIImageTool()],  # ✅ Adicione os parênteses para instanciar a ferramenta
            output_json=AnomalyJson, # ← This is how you specify that the output of this task should be parsed as JSON with a specific structure. The agent responsible for this task should return a JSON that matches the AnomalyJson model.
        )

    @task
    def localization_task(self) -> Task:
        return Task(
            config=self.tasks_config['localization_task'], # type: ignore[index]
            tools=[OpenAIImageTool()],
            output_json=LocalizationJson, # ← This is how you specify that the output of this task should be parsed as JSON with a specific structure. The agent responsible for this task should return a JSON that matches the LocalizationJson model.
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            context=[self.image_analysis_task()],
            output_json=RadiologyReport, # ← This is how you specify that the output of this task should be parsed as JSON with a specific structure. The agent responsible for this task should return a JSON that matches the RadiologyReport model.
        )
    
    @task
    def circle_radiography_task(self) -> Task:
        return Task(
            config=self.tasks_config["circle_radiography_task"], # type: ignore[index]
            context=[self.localization_task()],
            tools=[CircleTheAnomalyTool()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MedgemmaCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
