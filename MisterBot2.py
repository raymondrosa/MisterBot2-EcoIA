import os
import streamlit as st

# Importaciones alternativas más compatibles
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings  # Cambiado
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Cambiado
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate  # Cambiado
from langchain.schema import HumanMessage, AIMessage  # Cambiado

# Importación específica para ConversationalRetrievalChain
try:
    # Intento 1: Desde langchain (versión más reciente)
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    try:
        # Intento 2: Desde langchain_community
        from langchain_community.chains import ConversationalRetrievalChain
    except ImportError:
        try:
            # Intento 3: Desde langchain_experimental
            from langchain_experimental.chains import ConversationalRetrievalChain
        except ImportError:
            # Si todo falla, creamos nuestra propia clase simplificada
            st.error("No se pudo importar ConversationalRetrievalChain. Usando versión simplificada.")
            
            from langchain.chains.base import Chain
            from typing import Dict, List, Any
            
            class ConversationalRetrievalChain(Chain):
                """Versión simplificada de ConversationalRetrievalChain"""
                
                def __init__(self, llm, retriever, memory, combine_docs_chain_kwargs=None, **kwargs):
                    super().__init__()
                    self.llm = llm
                    self.retriever = retriever
                    self.memory = memory
                    self.combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
                    self.input_key = "question"
                    self.output_key = "answer"
                
                @property
                def input_keys(self) -> List[str]:
                    return [self.input_key]
                
                @property
                def output_keys(self) -> List[str]:
                    return [self.output_key]
                
                def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    question = inputs[self.input_key]
                    
                    # Recuperar documentos relevantes
                    docs = self.retriever.get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Obtener historial de chat
                    chat_history = ""
                    if self.memory:
                        memory_vars = self.memory.load_memory_variables({})
                        chat_history = memory_vars.get("chat_history", "")
                    
                    # Crear prompt
                    prompt_template = self.combine_docs_chain_kwargs.get("prompt", 
                        PromptTemplate.from_template("Contexto: {context}\n\nPregunta: {question}\n\nRespuesta:"))
                    
                    prompt = prompt_template.format(context=context, question=question, chat_history=chat_history)
                    
                    # Generar respuesta
                    response = self.llm.invoke(prompt)
                    
                    # Guardar en memoria
                    if self.memory:
                        self.memory.save_context({"question": question}, {"answer": response.content})
                    
                    return {self.output_key: response.content}
                
                @classmethod
                def from_llm(cls, llm, retriever, memory, combine_docs_chain_kwargs=None, **kwargs):
                    return cls(llm=llm, retriever=retriever, memory=memory, 
                              combine_docs_chain_kwargs=combine_docs_chain_kwargs, **kwargs)

# =============================
# CONFIGURACIÓN ECOIA
# =============================
MODEL_NAME = "llama3:8b-instruct-q4_0"
EMBED_MODEL = "nomic-embed-text"
PDF_PATH = "documento.pdf"
PERSIST_DIR = "memoria2/vector_db"
CHAT_MEMORY_FILE = "memoria2/chat.txt"
os.makedirs("memoria2", exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# =============================
# MEMORIA PERSISTENTE LIGERA (CORREGIDA)
# =============================
def cargar_memoria_txt(memory):
    """Carga el historial desde archivo de texto"""
    if not os.path.exists(CHAT_MEMORY_FILE):
        return
    
    try:
        with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Procesar líneas en pares (Usuario/Asistente)
        i = 0
        while i < len(lines):
            linea = lines[i].strip()
            if linea.startswith("Usuario:"):
                pregunta = linea.replace("Usuario:", "").strip()
                # Buscar la siguiente línea que sea del asistente
                if i + 1 < len(lines) and lines[i+1].strip().startswith("Asistente:"):
                    respuesta = lines[i+1].strip().replace("Asistente:", "").strip()
                    # Agregar a la memoria usando el método correcto
                    memory.save_context(
                        {"input": pregunta}, 
                        {"output": respuesta}
                    )
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    except Exception as e:
        print(f"Error al cargar memoria: {e}")

def guardar_memoria_txt(pregunta, respuesta):
    """Guarda el historial en archivo de texto"""
    try:
        with open(CHAT_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"Usuario: {pregunta}\n")
            f.write(f"Asistente: {respuesta}\n\n")
    except Exception as e:
        print(f"Error al guardar memoria: {e}")

# =============================
# CARGA DEL SISTEMA ECO
# =============================
@st.cache_resource
def cargar_sistema():
    if not os.path.exists(PDF_PATH):
        st.error(f"No se encuentra el archivo PDF: {PDF_PATH}")
        st.stop()

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
        db.persist()

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        output_key="answer"
    )
    cargar_memoria_txt(memory)

    QA_PROMPT = """Usa el siguiente contexto para responder la pregunta. 
    Si no sabes la respuesta, di que no tienes suficiente información. 
    Responde en español de forma clara y concisa.

    Contexto:
    {context}

    Pregunta: {question}
    Respuesta útil:"""

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(QA_PROMPT)},
        verbose=False
    )
    return qa_chain

# =============================
# INTERFAZ
# =============================
st.set_page_config(page_title="MisterBot2 - EcoIA", page_icon="⚡", layout="centered")

# =============================
# BARRA LATERAL CON LOGO (CORREGIDA)
# =============================
with st.sidebar:
    st.markdown("## 🤖 EcoIA")
    
    # Intentar cargar logo si existe (manejo de errores mejorado)
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=300)  # Ajusta el número según el tamaño que quieras
        except Exception as e:
            st.warning("Logo no disponible")
    else:
        st.markdown("---")
        st.markdown("### Sistema")
        st.info("✅ Optimizado 8GB RAM")
        st.markdown("### Modelos:")
        st.code("llama3:8b-instruct\nnomic-embed-text")
    
    st.markdown("---")
    st.markdown("### Comandos:")
    st.code("ollama pull llama3:8b-instruct-q4_0")
    st.code("ollama pull nomic-embed-text")

# =============================
# CONTENIDO PRINCIPAL
# =============================
st.title("⚡ MisterBot2 - EcoIA")
st.caption("Asistente ligero optimizado para 8GB RAM")

# Verificar que Ollama esté corriendo
try:
    qa_chain = cargar_sistema()
except Exception as e:
    st.error(f"Error al cargar el sistema: {str(e)}")
    st.info("Asegúrate de que Ollama esté corriendo y los modelos estén instalados:")
    st.code("ollama pull llama3:8b-instruct-q4_0\nollama pull nomic-embed-text")
    
    # Mostrar información de depuración
    with st.expander("Información de depuración"):
        import sys
        st.write("Python version:", sys.version)
        st.write("Paquetes instalados:")
        
        import pkg_resources
        installed_packages = [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
        for pkg in sorted(installed_packages):
            if 'langchain' in pkg.lower():
                st.write(f"- {pkg}")
    st.stop()

pregunta = st.text_input("Escribe tu pregunta:", placeholder="Ej: Resume el documento en 5 puntos clave...")

if pregunta:
    with st.spinner("Procesando..."):
        try:
            result = qa_chain.invoke({"question": pregunta})
            respuesta = result["answer"]
            st.markdown("### Respuesta")
            st.write(respuesta)
            guardar_memoria_txt(pregunta, respuesta)

            with st.expander("Historial reciente"):
                for msg in qa_chain.memory.chat_memory.messages:
                    if isinstance(msg, HumanMessage):
                        st.markdown(f"**Usuario:** {msg.content}")
                    elif isinstance(msg, AIMessage):
                        st.markdown(f"**EcoIA:** {msg.content}")
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>MisterBot2 EcoIA | Optimizado para bajo consumo | Prof. Raymond Rosa Ávila</div>",
    unsafe_allow_html=True
)