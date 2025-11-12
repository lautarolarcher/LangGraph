import os
from typing import TypedDict, Dict, List, Union
from langchain_core import messages
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

#ahora vamos a tener dos tipos de datos (mensjae humano y mensaje IA)
#defino el stado del Agente
class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

#defino el modelo del LLM

llm = ChatOpenAI(model="gpt-4o")

#defino las funciones

#aca en la funcion lo que hago es pasarle el estado actual del Agente
# le pido al LLM que me repsonda el mensaje de usuario, que esta en state["messages"] con invoke
# pero el mensaje del LLM si bien puede estar en el state porque lo defini con UNION todavia no esta almacednado ahi
# por eso al mesaje humano le agrego con append() el contenido del mensaje de AI Agent
def process (state:AgentState) -> AgentState:
    """Esta funcion resuelve la request del usuario."""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) #quiero agregarle a mis msj las rtas de la AI LLM

    print(f"\nAI:{response.content}")
    print(f"ESTADO ACTUAL: {state['messages']}")
    return state

#Graph

graph = StateGraph(AgentState)
#nodos
graph.add_node("process", process)
#edges
graph.add_edge(START, "process")
graph.add_edge("process", END)
#compliar
agent = graph.compile()

#POR FUERA DEL GRAPH
#voy a necesitar un historial de conversación para guarda lo que escribe el humano y lo que responda la IA

conversation_history = [] #inicializo el hisotial

#setear la interfaz de cuando corra el .py

user_input = input("Enter: ") #input() lo que hace es frenar el programa y mostrar "Enter: "
while user_input != "exit": #mientras el humano no diga "exit" doy las intrucciones de lo que tiene que hacer
    
    conversation_history.append(HumanMessage(content=user_input)) # por fuera del graph, junto lo que el humano viene preguntando

    result = agent.invoke({"messages" : conversation_history}) #defino la variable result = invocar el agente (graph compilado) 
                            #y le paso que el estado del AgentState omom "messages": el historial unido de lo que el humano pregunto 
                            # para que se ejecute process y responda 
                            # en process ya habiamos unido tambien a message la respuesta de la IA result ahora contiene: {"messages": [HumanMessage(...), AIMessage(...)]}
    conversation_history = result["messages"] # por eso aca se junta las preguntas y rtas, porque resul invoca a pocess y este devuelve mesagges
    
    user_input = input("Enter: ")
    

##EL HISTORIAL SE GUARDA EN LOCAL en un .TXT

with open("logging.txt", "w") as file: #with cierra automaticamente el archivo, 
    # open lo abre, #
    # w borra lo que tiene y comienza a escribir lo crea si no esta creado , 
    # as file asi llama a la variable
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n") #file.write(...): Escribe la cadena de texto especificada en el archivo.
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")