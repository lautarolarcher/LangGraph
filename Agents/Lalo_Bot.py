# Importaciones librerias

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Nuestro modelo para convertir texto en vectores (embeddings)

#FILE

TEXT_PATH = "Faqs_rapiboy_mexico.txt"
PERSIST_DIR = "./chroma_db_rapiboy"  # Define la carpeta local donde se guardará la base de datos de Chroma. El "./" significa "en la carpeta actual".
COLLECTION_NAME = "rapiboy_faqs"

loader = TextLoader(TEXT_PATH, encoding='utf-8') # Usamos TextLoader para archivos .txt

documents = loader.load() # Carga el contenido del archivo en la variable 'documents'

# Dividimos el texto en fragmentos (chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents) # Aplica la división de texto al contenido cargado. 'chunks' es una lista de fragmentos pequeños.


# === Creación de la Base de Datos Vectorial ===
if not os.path.exists(PERSIST_DIR): # Comprueba si la carpeta 'chroma_db_rapiboy' NO existe en el disco duro.
    os.makedirs(PERSIST_DIR)     # Si no existe, la crea.


print(f"Creando ChromaDB en {PERSIST_DIR}...")

# Crea la base de datos vectorial ChromaDB.
vectorstore = Chroma.from_documents(
    documents=chunks,     # Le pasa la lista de fragmentos de texto.
    embedding=embeddings,    # Le dice qué modelo usar para convertir esos textos en vectores numéricos.
    persist_directory=PERSIST_DIR,     # Le dice dónde guardar estos vectores en el disco duro.
    collection_name=COLLECTION_NAME     # Le da un nombre a la colección.
)

print("¡Base de datos vectorial ChromaDB creada con éxito!")
# Imprime un mensaje de confirmación final.

# Creamos el objeto 'retriever' que sabe cómo buscar
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5} # Devuelve los 5 fragmentos más relevantes
)

# STATE

class AgentState (TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


# Dict postulantes
    # esto seria una variable global 
applicant_data = {
    "nombre" : None, "apeliido" : None, "telefono" : None, "mail" : None,
    "RFC" : None, "INE" : None, "direccion" : None, "alcaldia/municipio" : None,
    "vehiculo" : None, "experiencia" : None
    }
list_faltante = list(applicant_data.keys())

# TOOLS

@tool
def faqs_retriever(query: str) -> str:
    """
    Esta herramienta busca y devuelve información relevante del documento de FAQs de Rapiboy Same Day.
    Úsala para responder preguntas sobre requisitos, pagos u horarios.
    """
    # Esta función se conecta a la DB, busca y devuelve el resultado.
    # No necesita redefinir PERSIST_DIR porque 'retriever' ya está configurado arriba.
    docs = retriever.invoke(query)
    
    if not docs:
        return "No se encontró información relevante en los FAQs."
    results = [f"Doc {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(results)

# --- Herramienta 2: start_registration (Iniciar Registro) ---
@tool
def start_registration(query: str) -> str:
    """
    Inicia el proceso de registro del chofer. Usa esta herramienta 
    cuando el usuario exprese su deseo de registrarse.
    """
    global list_faltante, applicant_data # Declaramos que usaremos las variables globales

    # Reiniciamos el proceso por si acaso (limpiamos datos de una sesión anterior)
    applicant_data = {k: None for k in applicant_data.keys()}
    list_faltante = list(applicant_data.keys())
    
    # Devolvemos el primer campo que necesitamos para que el agente lo pida
    first_field = list_faltante[0]
    return f"Registro iniciado. Por favor, proporciona tu {first_field}."


# --- Herramienta 3: submit_info (Enviar Información de un Campo) ---
@tool
def submit_info(info: str) -> str:
    """
    Envía la información proporcionada por el usuario para el campo actual.
    """
    global list_faltante, applicant_data # Declaramos que usaremos las variables globales
    
    if not list_faltante:
        return "Todos los datos han sido recolectados anteriormente. El registro está completo."

    current_field = list_faltante.pop(0) # Tomamos el primer campo que falta y lo removemos de la lista de faltantes
    applicant_data[current_field] = info # Guardamos la info en el diccionario global 'applicant_data'

    if not list_faltante:
        # Si ya no faltan campos, le decimos al agente que todo está OK
        return f"Gracias por tu {current_field}. Todos los datos están completos. Puedes usar la herramienta 'finish_registration' ahora."
    else:
        next_field = list_faltante[0]
        # Le pedimos el siguiente dato
        return f"Gracias por tu {current_field}. Ahora, por favor, ingresa tu {next_field}."


# --- Herramienta 4: finish_registration (Guardar Archivo Final) ---
@tool
def finish_registration(filename: str = "registro_rapiboy.txt") -> str:
    """Guarda todos los datos del registro en un archivo de texto."""
    global applicant_data # Usamos la variable global
    try:
        with open(filename, 'w') as f:
            for key, value in applicant_data.items():
                f.write(f"{key.capitalize()}: {value}\n")
        return f"Registro guardado exitosamente en {filename}. ¡Gracias por postularte!"
    except Exception as e:
        return f"Error al guardar el registro: {e}"


# =========================================================
# === Parte 4: Definición del Modelo y Herramientas ===
# =========================================================

# Unimos todas las herramientas disponibles al modelo de IA
all_tools = [faqs_retriever, start_registration, submit_info, finish_registration]
llm_with_tools = llm.bind_tools(all_tools) # Enlaza las herramientas al LLM

# =========================================================
# === Parte 5: Lógica de los Nodos del Grafo ===
# =========================================================

# --- Nodo 1: our_agent (El Cerebro Principal) ---
def our_agent(state: AgentState) -> AgentState:
    """Nodo que invoca al LLM para decidir si responder o usar herramientas."""
    print("🧠 AGENTE: Pensando...")
    system_prompt = SystemMessage(content="""
    Eres un asistente de Rapiboy Same Day. Tu objetivo es responder preguntas sobre la operación 
    o ayudar al usuario a registrarse.

    Instrucciones Clave:
    1. Responde preguntas generales usando la herramienta 'faqs_retriever'.
    2. Si el usuario dice "quiero registrarme", usa la herramienta 'start_registration'.
    3. Si el usuario está en proceso de registro, usa 'submit_info' para enviar la información que te dé.
    4. Cuando todos los datos estén completos, usa 'finish_registration' para guardar el archivo.
    """)
    
    messages = [system_prompt] + list(state['messages']) # Combina el prompt del sistema con el historial de mensajes
    response = llm_with_tools.invoke(messages) # Invoca al LLM
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 AGENTE: Decidió usar herramienta: {[tc['name'] for tc in response.tool_calls]}")
    
    return {'messages': [response]} # Devuelve solo la nueva respuesta del LLM (el middleware se encarga de añadirla al historial)


# --- Nodo 2: should_continue (El Router de Decisión Condicional) ---
def should_continue(state: AgentState) -> str:
    """Determina si el agente debe ir a ejecutar herramientas o terminar."""
    last_message = state['messages'][-1]
    
    if not last_message.tool_calls:
        # Si NO hay tool_calls (respuesta de texto final), el flujo termina
        print("➡️ AGENTE: Terminando conversación (respuesta final).")
        return "end"
    else:
        # Si SÍ hay tool_calls, el flujo continúa al nodo de herramientas
        return "continue" # CORREGIDO: Añadido 'return "continue"'

# =========================================================
# === Parte 6: Construcción del Grafo (Flowchart) ===
# =========================================================

graph = StateGraph(AgentState)
graph.add_node("our_agent", our_agent) # Añade el nodo 'our_agent' (IA principal)
graph.add_node("tools", ToolNode(all_tools)) # Añade el nodo 'tools' (Ejecutor de funciones)

graph.set_entry_point("our_agent") # El inicio del grafo es 'our_agent'

# Define la lógica condicional que sale de 'our_agent'
graph.add_conditional_edges(
    "our_agent",
    should_continue, # Usa la función para decidir el camino
    {"continue": "tools", "end": END} # Si 'continue', va a 'tools'; si 'end', termina.
)

graph.add_edge("tools", "our_agent") # Después de ejecutar tools, siempre vuelve a 'our_agent' para que la IA lea el resultado.

app = graph.compile() # Compila el grafo completo en un objeto ejecutable 'app'.


# =========================================================
# === Parte 7: Interfaz de Usuario (UI) y Ejecución ===
# =========================================================
def run_agent_in_terminal():
    """Función principal para correr el agente en la terminal."""
    print("\n\n============= INICIO DE ASISTENTE RAPIBOY =============")
    print("Hola, soy tu asistente virtual de Rapiboy Same Day. ¿En qué puedo ayudarte?")
    
    inputs = {"messages": []} # Inicializa el historial de mensajes vacío
    
    while True:
        user_input = input("\n👤 TÚ: ") # Pide input al usuario
        if user_input.lower() in ["exit", "quit", "salir"]:
            break # Si escribe salir, termina el bucle
        
        inputs["messages"].append(HumanMessage(content=user_input)) # Agrega el mensaje humano al historial
        
        result = app.invoke(inputs) # Ejecuta el agente con el input actual
        inputs["messages"] = result["messages"] # Sincroniza el historial local con el resultado completo del agente

        # Imprimimos la respuesta final o el resultado de la herramienta de forma legible
        final_response = inputs["messages"][-1]
        if isinstance(final_response, AIMessage):
            print(f"\n🤖 AI: {final_response.content}")
        elif isinstance(final_response, ToolMessage):
             print(f"\n🛠️ HERRAMIENTA RESULTADO: {final_response.content}")

        
    print("\n============= FIN DE CONVERSACIÓN =============")


if __name__ == "__main__":
    run_agent_in_terminal()
    # Ejecuta la función principal solo si el script se está corriendo directamente, no si se importa.






