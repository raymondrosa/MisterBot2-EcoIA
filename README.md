Licencia: CC-NC; Código de registro: 2602234666960; Prof. Raymond Rosa Ávila

🤖 MisterBot2 – Eco-IA

Asistente conversacional inteligente con memoria persistente, RAG (Retrieval-Augmented Generation) y procesamiento de documentos PDF utilizando modelos locales con Ollama.

Construido con:

Streamlit (Interfaz web)

LangChain (Orquestación LLM)

Ollama (Modelos locales)

ChromaDB (Vector Store persistente)

Embeddings locales

Memoria conversacional persistente en archivo .txt

🚀 Características

✅ Modelos 100% locales (sin depender de APIs externas)

✅ Sistema RAG sobre documentos PDF

✅ Vectorización persistente con Chroma

✅ Memoria de conversación guardada en archivo

✅ Interfaz profesional con fondo personalizado y logo

✅ Optimizado para bajo consumo (modelo cuantizado)

🧠 Arquitectura del Sistema

Usuario → Streamlit UI
↓
ConversationalRetrievalChain
↓
LLM (Ollama - llama3)
↓
Chroma Vector Store
↓
Embeddings (nomic-embed-text)
↓
PDF Indexado

Memoria adicional:

memoria2/chat.txt

memoria2/vector_db/

📦 Requisitos Previos

Antes de instalar, asegúrate de tener:

Python 3.9 o superior

Ollama instalado

Git (opcional, pero recomendado)

🛠️ Instalación Paso a Paso
1️⃣ Clonar el repositorio
git clone https://github.com/TU-USUARIO/MisterBot2.git
cd MisterBot2
2️⃣ Crear entorno virtual

Windows:

python -m venv venv
venv\Scripts\activate

Mac / Linux:

python3 -m venv venv
source venv/bin/activate
3️⃣ Instalar dependencias

Crear archivo requirements.txt con:

streamlit
langchain
langchain-community
langchain-experimental
chromadb
ollama
pypdf

Luego instalar:

pip install -r requirements.txt
4️⃣ Instalar Ollama

Descargar desde:

https://ollama.com

Verificar instalación:

ollama --version
5️⃣ Descargar los modelos necesarios

Tu aplicación usa:

llama3:8b-instruct-q4_0

nomic-embed-text

Instalarlos con:

ollama pull llama3:8b-instruct-q4_0
ollama pull nomic-embed-text

⚠️ Este paso es obligatorio.

6️⃣ Verificar estructura del proyecto

Tu carpeta debe contener:

MisterBot2.py
documento.pdf
fondo.png
logo.png
memoria2/

Si no existe memoria2, el sistema la crea automáticamente.

7️⃣ Ejecutar la aplicación
streamlit run MisterBot2.py

Luego abrir en el navegador:

http://localhost:8501
📁 Estructura del Proyecto
MisterBot2/
│
├── MisterBot2.py
├── documento.pdf
├── fondo.png
├── logo.png
├── requirements.txt
└── memoria2/
    ├── chat.txt
    └── vector_db/
🧠 ¿Cómo Funciona?

Si existe base vectorial → la reutiliza.

Si no existe → indexa el PDF automáticamente.

Carga memoria previa desde chat.txt.

Cada pregunta:

Recupera contexto relevante

Genera respuesta

Guarda conversación

Mantiene ventana de memoria configurable (k=5).

🧪 Solución de Problemas
❌ Error: No se pudo importar ConversationalRetrievalChain

La app incluye fallback automático para versiones distintas de LangChain.
Si persiste el error:

pip install --upgrade langchain langchain-community
❌ Error: Ollama no está corriendo

Ejecutar:

ollama serve
❌ No encuentra el PDF

Verifica que el archivo se llame exactamente:

documento.pdf
⚙️ Personalización

Puedes modificar en el código:

MODEL_NAME = "llama3:8b-instruct-q4_0"
EMBED_MODEL = "nomic-embed-text"

También puedes ajustar:

Tamaño de chunks

Número de documentos recuperados (k)

Ventana de memoria conversacional

🔐 Licencia

Licencia Creative Commons CC-NC
Autor: Prof. Raymond Rosa Ávila

🌎 Futuras Mejoras (Roadmap)

Soporte para múltiples PDFs

Memoria estructurada en JSON

Deploy en servidor Linux

Dockerización

Integración con GitHub Pages (frontend ligero)

Autenticación multiusuario

Dashboard analítico de consultas

🧭 Visión

MisterBot2 no es solo un chatbot.
Es un núcleo cognitivo local, soberano y escalable.
Una plataforma base para asistentes especializados en ingeniería, educación e investigación científica.

El siguiente paso natural: contenerizarlo y desplegarlo en infraestructura cloud híbrida.
