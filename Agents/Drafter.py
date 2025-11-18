#CONSIGA

# ¡Nuestra empresa no está trabajando de forma eficiente! Dedicamos demasiado tiempo a
# redactar documentos, ¡y esto tiene que solucionarse!

# Para la empresa, se necesita crear un Sistema de Agentes de IA que pueda
# agilizar la redacción de documentos, correos electrónicos, etc. El Sistema de Agentes de IA
# debe permitir la colaboración humano-IA, es decir, que el humano pueda
# proporcionar retroalimentación continua y que el agente de IA
# detenga el proceso cuando el humano esté satisfecho con el borrador. El sistema también debe
# ser rápido y permitir guardar los borradores.



#Librery
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.pregel.main import SyncQueue

load_dotenv()

document_content = ""
#Esta es una variable global. Se podria haber usado  "inyectar un estado" pero es mas complejo (investigar)

#State

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages] #Uso annotated para especificar el tipo de dato y add_messages para que actualice el estado y no lo reempplace


#Tools
@tool
def update(content: str) -> str:
    """Actualiza el documento con el contenido proporcionado."""
    global document_content
    document_content = content
    return f"El documento ha sido actualizado exitosamente! El contenido actual es:\n{document_content}"

@tool #adorno que convierte en una herramienta que la IA puede llamar

def save(filename: str) -> str:
    """Guarda el actual documento como un archivo de texto y finaliza el proceso.
    
    Argumento:
        filename: Nombre para el siguiente archivo.
    """
    #la docstrig es clave para que la IA sepa a que tool llamar

    global document_content # Crucial. Le dice a Python: "No crees una variable local document_content, 
                            # por favor usa la variable global que definí arriba". 
                            # Permite modificar la variable global desde dentro de la función.

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try: #manejo de erroes:
            #le dice a Python: "Intenta ejecutar este código; si falla, no detengas el programa, sino ve a este otro bloque de código para manejar la situación."
            #
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 El documento ha sido guardado como: {filename}")
        return f"El documento ha sido guardado exitosamente como '{filename}'."
    
    except Exception as e: 
            # except Exception as e: Este bloque atrapa cualquier tipo de error (Exception) que haya ocurrido en el try. Guarda la información del error en una variable temporal llamada e (puedes llamarla como quieras, como error_info).
            # return f"Error saving document: {str(e)}": En lugar de detener el programa, la función devuelve un mensaje de error amigable al usuario (y a la IA), que incluye la descripción técnica del error (str(e)).
        return f"Error al guarda el documento: {str(e)}"
    

tools = [update, save]


#Model   

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

#funciones del graph

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    Eres un redactor, ayudaras al ususario a actualizar y modificar sus documentos.
    
    - Si el usuario quiere actualizar o modificar el contenido, usa la tool 'update' tool con el contenido actualizado.
    - Si el usuario quiere guardar y finalizar, necesitas usar la tool 'save'. 
    - Asegurate siempre de mostrar el actual estado del documento despues de las modificaciones.
    
    El actual contenido del documento es:{document_content}
    """)

    if not state["messages"]: #una lista vacia es False, por lo tanto if not lista vacia seria algo not state["messages"] significa not False, que es True
        user_input = "Estoy listo para ayudarte a actualizar un documento. ¿Qué te gustaría crear?"
        user_message = HumanMessage(content=user_input)
        #este va a ser el primer mensaje para la primera vuelta donde 
        #no hay mensajes y el HummanMessage = user_messge es el unico que se va a usar en all_messages

    else:
        user_input = input("\n¿Qué desea hacer con el documento?")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls: #hasattr verifica si un objeto dado tiene un atributo o propiedad específico.
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determina si debemos continuar o terminar la conversación."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Función que imprime los mensajes en un formato más legible"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

#Graph

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


#por fuera 

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()