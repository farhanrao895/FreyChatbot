import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import tempfile
import multiprocessing
from langchain.schema import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Page config
st.set_page_config(page_title="Document Reader New", page_icon="📘", layout="centered")
load_dotenv()

# Config
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-pro-latest"
CHUNK_SIZE = 12500
CHUNK_OVERLAP = 2000
MAX_CHUNKS = 300
BATCH_SIZE = 5
QUOTA_LIMIT = 150

def extract_text_with_ocr(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
    except Exception as e:
        return None, f"OCR failed for {os.path.basename(pdf_path)}: {str(e)}"
    return text, None

def process_pdf(pdf_file):
    try:
        loader = PyMuPDFLoader(pdf_file)
        docs = loader.load()
        if all(not doc.page_content.strip() for doc in docs):
            ocr_text, ocr_error = extract_text_with_ocr(pdf_file)
            if ocr_text and ocr_text.strip():
                return [Document(page_content=ocr_text, metadata={"source": pdf_file})], True
            else:
                return f"{os.path.basename(pdf_file)} - {ocr_error}", False
        return docs, True
    except Exception as e:
        return f"{os.path.basename(pdf_file)} - {str(e)}", False

def process_pdfs_in_folder(folder_path):
    folder_path = os.path.normpath(folder_path)  # ✅ Normalize path

    if not os.path.exists(folder_path):
        raise ValueError(f"🚫 Folder does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"🚫 Not a directory: {folder_path}")

    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDFs found in folder.")

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("📂 Initializing document processing..."):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_pdf, pdf_files)

        documents = []
        for i, (result, success) in enumerate(results):
            status_text.markdown(f"📄 Processing {i+1}/{len(pdf_files)}...")
            if success:
                documents.extend(result)
            else:
                st.warning(f"Skipped: {result}")
            progress_bar.progress((i+1)/len(pdf_files))

    progress_bar.empty()
    return documents

def process_text_chunks(documents):
    text = "\n".join(doc.page_content for doc in documents if doc.page_content.strip())
    if not text.strip():
        raise ValueError("No extractable text found in documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("Failed to generate chunks.")
    return chunks

def embed_text_in_batches(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = None
    delay = 60 / (QUOTA_LIMIT / BATCH_SIZE)

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔧 Creating embeddings..."):
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            status_text.markdown(f"⚙️ Processing batch {i//BATCH_SIZE+1}")
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    temp = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(temp)
                time.sleep(delay)
            except Exception as e:
                st.error(f"❌ Embedding error: {str(e)}")
                break
            progress_bar.progress(min((i + BATCH_SIZE)/len(chunks), 1.0))

    progress_bar.empty()
    return vector_store

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": MAX_CHUNKS})

    system_template = "You are a helpful assistant. Use document context if relevant.\n{chat_history}"
    human_template = "{question}\n\nRelevant document context:\n{context}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )

def initialize_session_state():
    for key in ["vector_store", "messages", "conversation_chain"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "messages" else None

def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def main():
    st.markdown("## 📚 <span style='color:#2c3e50'>Document Reader New</span>", unsafe_allow_html=True)
    st.markdown("---")
    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Settings")
        folder_path = st.text_input("📁 Enter PDF folder path:")
        st.caption(f"Path will be normalized: `{os.path.normpath(folder_path)}`")

        if st.button("🚀 Process Documents"):
            if folder_path:
                try:
                    with st.status("🔄 Processing...", expanded=True):
                        st.write("📥 Reading PDFs...")
                        docs = process_pdfs_in_folder(folder_path)
                        st.success(f"✅ Loaded {len(docs)} PDFs.")

                        st.write("✂️ Splitting text...")
                        chunks = process_text_chunks(docs)

                        st.write("📌 Generating embeddings...")
                        vector_store = embed_text_in_batches(chunks)

                        st.session_state.vector_store = vector_store
                        st.session_state.conversation_chain = get_conversational_chain()
                        st.success(f"🔍 Ready! Processed {len(chunks)} chunks.")

                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
            else:
                st.warning("📂 Please enter a valid folder path.")

        st.markdown("---")
        if st.session_state.vector_store:
            st.metric("Stored Chunks", st.session_state.vector_store.index.ntotal)
            st.metric("Chunk Size", f"{CHUNK_SIZE} chars")
            st.metric("Batch Size", BATCH_SIZE)

        st.markdown("---")
        if st.button("🧹 Clear Chat Memory"):
            st.session_state.messages = []
            if st.session_state.conversation_chain:
                st.session_state.conversation_chain.memory.clear()
            st.rerun()

    display_chat_history()

    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store and st.session_state.conversation_chain:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.conversation_chain({"question": prompt})
                        response = result["answer"]
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)

                        if "source_documents" in result:
                            with st.expander("📎 References"):
                                for i, doc in enumerate(result["source_documents"][:5]):
                                    st.markdown(f"**Doc {i+1}**\n\n```\n{doc.page_content[:500]}...\n```")
                    except Exception as e:
                        err = f"🚨 Error: {str(e)}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            msg = "⚠️ Please process documents first."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Powered by Gemini 1.5 Pro • Local Secure Processing</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
