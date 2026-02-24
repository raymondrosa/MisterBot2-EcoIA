# ============================================
# MISTERBOT2 - ECO IA (VERSIÓN BLINDADA)
# Compatible con múltiples versiones LangChain
# ============================================

import os
import base64
import streamlit as st

# =============================
# IMPORTACIONES LANGCHAIN
# =============================
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Import robusto del chain
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    try:
        from langchain_community.chains import ConversationalRetrievalChain
    except ImportError:
        from langchain_experimental.chains import ConversationalRetrievalChain


# ============================================
# CONFIGURACIÓN
# ============================================

MODEL_NAME = "llama3:8b-instruct-q4_0"
EMBED_MODEL = "nomic-embed-text"
PDF_PATH = "documento.pdf"
PERSIST_DIR = "memoria2/vector_db"
CHAT_MEMORY_FILE = "memoria2/chat.txt"

os.makedirs("memoria2", exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)


# ============================================
# FUNCIONES MEMORIA TXT
# ============================================

def cargar_memoria_txt(memory):
    if not os.path.exists(CHAT_MEMORY_FILE):
        return

    with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("Usuario:"):
            pregunta = lines[i].replace("Usuario:", "").strip()
            if i + 1 < len(lines) and lines[i+1].startswith("Asistente:"):
                respuesta = lines[i+1].replace("Asistente:", "").strip()

                try:
                    memory.save_context(
                        {"input": pregunta},
                        {"output": respuesta}
                    )
                except:
                    pass

                i += 2
            else:
                i += 1
        else:
            i += 1


def guardar_memoria_txt(pregunta, respuesta):
    with open(CHAT_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"Usuario: {pregunta}\n")
        f.write(f"Asistente: {respuesta}\n\n")


# ============================================
# SISTEMA RAG BLINDADO
# ============================================

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

    # Usamos output_key estándar seguro
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
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


# ============================================
# CONFIG STREAMLIT
# ============================================

st.set_page_config(
    page_title="MisterBot2 - EcoIA",
    page_icon="⚡",
    layout="centered"
)


# ============================================
# ESTILO VISUAL PROFESIONAL
# ============================================

def set_background_local(image_path):

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        st.markdown(f"""
        <style>

        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-attachment: fixed;
        }}

        .main .block-container {{
            background-color: rgba(0,0,0,0.85);
            padding: 2rem;
            border-radius: 20px;
            backdrop-filter: blur(8px);
        }}

        h1, h2, h3 {{
            color: #00ffe0 !important;
        }}

        p, label {{
            color: #f5f5f5 !important;
            font-size: 17px;
        }}

        .stTextInput input {{
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            border: 2px solid #00ffe0 !important;
        }}

        .user-bubble {{
            background-color: #004d40;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            color: white;
        }}

        .ai-bubble {{
            background-color: #1a1a1a;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #00ffe0;
            color: white;
        }}

        pre {{
            background-color: #111 !important;
            color: #00ff00 !important;
            border-radius: 10px;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.9);
        }}

        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"No se pudo cargar fondo: {e}")


set_background_local("fondo.png")


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## 🤖 Eco-IA")

    if os.path.exists("logo.png"):
        st.image("logo.png", width=280)
    else:
        st.info("Sistema Optimizado")

    st.markdown("---")
    st.markdown("### Modelos:")
    st.code("llama3:8b-instruct-q4_0\nnomic-embed-text")


# ============================================
# INTERFAZ PRINCIPAL
# ============================================

st.title("🤖 MisterBot2 - EcoIA")
st.caption("Sistema RAG con Memoria Persistente (Versión Blindada)")

try:
    qa_chain = cargar_sistema()
except Exception as e:
    st.error(f"Error al cargar sistema: {e}")
    st.stop()

pregunta = st.text_input("Escribe tu pregunta:")

if pregunta:
    with st.spinner("Procesando..."):
        try:
            result = qa_chain.invoke({"question": pregunta})

            # 🔐 Blindaje automático de clave
            if "result" in result:
                respuesta = result["result"]
            elif "answer" in result:
                respuesta = result["answer"]
            else:
                respuesta = str(result)

            guardar_memoria_txt(pregunta, respuesta)

            st.markdown("### Respuesta")
            st.markdown(
                f"<div class='ai-bubble'>{respuesta}</div>",
                unsafe_allow_html=True
            )

            with st.expander("Historial reciente"):
                for msg in qa_chain.memory.chat_memory.messages:
                    if isinstance(msg, HumanMessage):
                        st.markdown(
                            f"<div class='user-bubble'><strong>Usuario:</strong><br>{msg.content}</div>",
                            unsafe_allow_html=True
                        )
                    elif isinstance(msg, AIMessage):
                        st.markdown(
                            f"<div class='ai-bubble'><strong>EcoIA:</strong><br>{msg.content}</div>",
                            unsafe_allow_html=True
                        )

        except Exception as e:
            st.error(f"Error al procesar pregunta: {e}")


st.markdown("---")
st.markdown(
    "<span style='color:#00ffe0;font-weight:bold;'>Licencia CC-NC | Prof. Raymond Rosa Ávila</span>",
    unsafe_allow_html=True
)