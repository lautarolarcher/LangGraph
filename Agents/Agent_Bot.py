from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv #se usa para almacenar las API KEY secretas

load_dotenv()

#defino el estado
class AgentState(TypedDict):
    message: List[HumanMessage]

#invocar la LLM
llm = ChatOpenAI(model = "gpt-4o")

#defino las funciones
def process(state:AgentState) -> AgentState:
    respuesta = llm.invoke(state["message"])
    print(f"\nAI: {respuesta.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


user_input = input("Enter: ") #detiene la ejecución del programa y muestra el mensaje que puse como argumento
contador=0
while user_input != "exit": # mientras se cumpla la condicion
    contador+=1
    agent.invoke({"message": [HumanMessage(content=user_input)]}) # que debe hacer mienstras se cumpla la condicion
    print(f"respuesta {contador}") 
    user_input = input("\nEnter next: ") # Una vez que el agente de IA ha respondido, esta línea vuelve a pedirle al usuario que escriba algo nuevo.
                                       # La variable user_input se actualiza con el nuevo texto que escribas.

