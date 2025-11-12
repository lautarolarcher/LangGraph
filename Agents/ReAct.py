from email import message
from typing import Annotated, Sequence, TypedDict #Anotated:Permite añadir metadatos o instrucciones adicionales a un tipo existente, Sequence: Representa una lista ordenada e inmutable de elementos
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import tool_call
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tool_node

load_dotenv()

# State
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    # Sequence[BaseMessage]: El tipo de datos esperado. Le dice a Python que messages debe ser una lista/secuencia que contenga objetos de tipo BaseMessage (mensajes de LangChain).
    # add_messages: Esto es el metadato. Es una función redutora (agrega y no reemplza los datos en el State) de LangGraph que actúa como un middleware.
    # Funcionamiento de Annotated aquí: Cuando LangGraph pasa datos a través del grafo, 
    # si ve esta anotación, no trata messages como una simple lista. 
    # En su lugar, usa la función add_messages para fusionar inteligentemente los mensajes nuevos con los viejos, 
    # en lugar de sobrescribir toda la lista. Es clave para mantener el historial de conversación.


#tools

@tool #decorador
def add (a=int, b=int):
    """Esta tool suma a + b, siendo a y b los enteros ingresados."""
    return a + b

@tool #decorador
def substract (a=int, b=int):
    """Esta tool resta a - b, siendo a y b los enteros ingresados."""
    return a - b

@tool #decorador
def multiply (a=int, b=int):
    """Esta tool multiplica a * b, siendo a y b los enteros ingresados."""
    return a * b  

tools =[add, substract, multiply]

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
    #aca con blid_tools le digo al modelo de gpt que tiene tools para usar
    #y le paso que tools puede usar

# Functions
def model_call (state:AgentState) -> AgentState: # aca le paso el estado (de tipo AgentState) y me tiene que devolver el estado tmb
    """Esta funcion llama al modelo LLM."""
    system_prompt = SystemMessage(content=""""Sos mi asistente de IA, responde mi consulta de la mejor 
        manera que te permitan tus habilidades""") # aca crea el mensaje para dar las instrucciones del tipo SystemMessage
    response = llm.invoke([system_prompt] + state["messages"]) #aca al mensaje del sistema le agrego el mensaje humano y ese va a ser todo el INPUT para la LLM
    return{"messages" : response} #Devuelve un diccionario (que cumple con la estructura AgentState), podria haber sido return state

    # ¿Dónde está el historial viejo? ¡Aquí es donde entra en juego Annotated y add_messages!
    # LangGraph, al ver la anotación Annotated[..., add_messages] en tu AgentState, 
    # sabe que debe tomar esta nueva respuesta y añadirla al historial existente, sin borrarlo.


def should_continue (state:AgentState) -> str:
    """Esta funcion decide si usar una tool o salir del bucle."""
    messages = state["messages"]
    last_message = messages[-1]
    # Esta función decide el siguiente paso. 
    # Revisa si el último mensaje que genero la IA (last_message) tiene llamadas a herramientas (tool_calls).
    # Esto es posible ya que con blind_tools() la IA con el input que le paso puede llamar tools.
    
    # Comprueba si el último mensaje contiene una "tool_call"
    if not last_message.tool_calls:
       return "end_edge"
    else:
       return "continue"


# Graph

graph = StateGraph(AgentState)
#nodes

graph.add_node("our_agent", model_call)
#nodes tools

    #aca defino la variable, que contiene toda la lógica configurada para ejecutar las herramientas.
tool_node = ToolNode(tools=tools) # ToolNode nodo predefinido que sirve para leer las tools que defini
graph.add_node("tools", tool_node)

#edges
graph.add_edge(START,"our_agent") #set_entry_point("...") se usa cunado le quiero dar un nombre especifico al nodo inicial
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end_edge" : END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Suma 40 + 12 y al reusltado de esa suma restale 10, al resultado de la resta multiplicalo por 100. Despues decime la capital de Canada.")]}
print_stream(app.stream(inputs, stream_mode="values"))

# def print_stream(stream):
# Define una función que espera un argumento llamado stream.
# ¿Qué es stream? Es un generador (un tipo especial de iterable en Python) que devuelve el estado del agente en cada paso del proceso (cada vez que un nodo se ejecuta).
# for s in stream::
# Itera sobre cada paso (s) que el agente va generando. s es el diccionario de AgentState en ese momento (ej. {"messages": [...]}).
# message = s["messages"][-1]:
# En cada paso, extrae el último mensaje de la lista de mensajes. Esto nos permite ver la acción más reciente que acaba de ocurrir (la respuesta de la IA, una llamada a herramienta, o el resultado de una herramienta).
# if isinstance(message, tuple)::
# Comprueba si el mensaje es una tupla (que es la forma abreviada de definir un mensaje que usamos en la entrada inputs).
# Si es una tupla (ej. ('user', 'Hola')), simplemente lo imprime tal cual: print(message).
# else: / message.pretty_print():
# Si no es una tupla, es un objeto completo de LangChain (HumanMessage, AIMessage, ToolMessage, etc.).
# Estos objetos tienen un método incorporado llamado .pretty_print() que los formatea de una manera visualmente agradable en la consola, mostrando el tipo de mensaje y su contenido.


