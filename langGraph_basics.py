# LLM
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")



from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END  
#clase especial para describir la forma de un diccionario (qué claves tiene y de qué tipo es cada valor).


####Prompt Chaining####

# Graph state

#Acá se hereda de TypedDict para decir: “mi estado es un dict con estas claves y tipos”.
class State(TypedDict): 
    palabra1: str
    palabra2: str
    palabra3: str
    rima: str

# Nodos (cada función recibe y devuelve estado o parte del estado)

#(state: State) = parámetro llamado state que debería cumplir el “molde” State.
def palabra1(state: State):
    """Primer LLM call para obtener la palabra1"""
     msg = llm.invoke(f"nventa una palabra: {state['palabra1']}.")
     return({"palabra1":msg.content})

def palabra2(state: State):
    """Segundo LLM call para obtener la palabra2"""
    msg = llm.invoke(f"Inventa una palabra que rime con: {state['palabra1']} y no sea la misma que {state['palabra1']}.")
    return({"palabra2": msg.content})

def palabra3(state: State):
    """Tercer LLM call para obtener la palabra3"""
    msg = llm.invoke(f"Inventa una palabra que rime con: {state['palabra1']} y {state['palabra2']}.")
    return({"palabra3": msg.content})

def rima(state: State):
    """Cuarto LLM call para obtener la rima"""
    msg = llm.invoke(f"Inventa una rima para las palabras: {state['palabra1']}, {state['palabra2']}, {state['palabra3']}.")
    return({"rima": msg.content})

#Se devuelve un diccionario parcial de estado con la clave que actualizaste.
#LangGraph fusiona lo que devuelven los nodos con el estado global 
#(por eso suele usarse Annotated[...] con reducers cuando varios nodos escriben a la misma clave; 
#acá cada nodo escribe a una clave distinta).

def check_rima(state: State):
    if state['palabra1'] == state['palabra2'] or state['palabra1'] == state['palabra3'] or state['palabra2'] == state['palabra3']:

        return("rima": "Las palabras no riman.")
    else:
        return("rima": "Las palabras riman.")

#Graph definition  

workflow = StateGraph(State)

workflow.add_node("palabra1", palabra1)
workflow.add_node("palabra2", palabra2)
workflow.add_node("palabra3", palabra3)
workflow.add_node("rima", rima)
workflow.add_node("check_rima", check_rima)

workflow.add_edge(START, "palabra1")
workflow.add_edge("palabra1", "palabra2")
workflow.add_edge("palabra2", "palabra3")
workflow.add_edge("palabra3", "rima")
workflow.add_edge("rima", "check_rima")
workflow.add_edge("check_rima", END)

chain = workflow.compile()

state = chain.invoke({"palabra1": "cielo", "palabra2": "hielo", "palabra3": "fiero"})
print(state)

