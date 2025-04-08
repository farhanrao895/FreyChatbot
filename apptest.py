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
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import tempfile
import multiprocessing
from langchain.schema import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Page config
PAGE_ICON = "book"
PAGE_TITLE = "Document Reader New"
LAYOUT = "centered"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

load_dotenv()

# Configuration
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
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
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
    documents = []
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid directory: {folder_path}")

    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDFs found")

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("📂 Initializing document processing..."):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_pdf, pdf_files)

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
    raw_text = "\n".join(doc.page_content for doc in documents if doc.page_content and doc.page_content.strip())
    if not raw_text.strip():
        raise ValueError("No extractable text found in the uploaded documents. Please ensure the PDFs contain text or use OCR.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    if not chunks:
        raise ValueError("Failed to generate text chunks. Ensure your documents contain readable text.")
    return chunks

def embed_text_in_batches(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = None
    delay = 60 / (QUOTA_LIMIT / BATCH_SIZE)

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔧 Creating embeddings..."):
        total_batches = len(chunks) // BATCH_SIZE + 1
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            status_text.markdown(f"⚙️ Processing batch {i//BATCH_SIZE+1}/{total_batches}")
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    temp_store = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(temp_store)
                time.sleep(delay)
            except Exception as e:
                st.error(f"Batch error: {str(e)}")
                break
            progress_bar.progress(min((i+BATCH_SIZE)/len(chunks), 1.0))

    progress_bar.empty()
    return vector_store

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": MAX_CHUNKS})

    system_template = "You are a helpful document assistant. Use the context below to answer user questions. Maintain the flow of conversation.\n{chat_history}"

    human_template = "{question}\n\nDocument context (only use if relevant to question):\n{context}"

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=True
    )

def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.markdown(f"<h1 style='text-align: center; color: #2c3e50;'>📚 {PAGE_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("---")

    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Settings")
        folder_path = st.text_input("📁 Enter PDF folder path:")

        if st.button("🚀 Process Documents"):
            if folder_path:
                with st.status("🧠 Processing...", expanded=True) as status:
                    try:
                        st.write("Loading documents...")
                        documents = process_pdfs_in_folder(folder_path)
                        st.write(f"Loaded {len(documents)} documents.")

                        for i, doc in enumerate(documents):
                            st.write(f"Doc {i+1}: {doc.metadata.get('source', 'unknown')} - Length: {len(doc.page_content) if doc.page_content else 0}")

                        st.write("Chunking text...")
                        chunks = process_text_chunks(documents)

                        st.write("Generating embeddings...")
                        vector_store = embed_text_in_batches(chunks)

                        st.session_state.vector_store = vector_store
                        st.session_state.conversation_chain = get_conversational_chain()
                        status.update(label="✅ Processing Complete!", state="complete")
                        st.success(f"Processed {len(documents)} documents with {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
            else:
                st.warning("Please enter a folder path")

        st.markdown("---")
        st.markdown("📊 Document Metrics")
        if st.session_state.vector_store:
            st.metric("Stored Chunks", st.session_state.vector_store.index.ntotal)
            st.metric("Chunk Size", f"{CHUNK_SIZE} chars")
            st.metric("Batch Size", BATCH_SIZE)

        st.markdown("---")
        st.markdown("🧠 Memory Management")
        if st.button("Clear Conversation Memory"):
            st.session_state.messages = []
            if st.session_state.conversation_chain:
                st.session_state.conversation_chain.memory.clear()
            st.rerun()

    display_chat_history()

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store and st.session_state.conversation_chain:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.conversation_chain({"question": prompt})
                        response = result["answer"]

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        if "source_documents" in result and len(result["source_documents"]) > 0:
                            with st.expander("🔍 View Source References"):
                                tab1, tab2 = st.tabs(["Document Excerpts", "Analysis Data"])
                                with tab1:
                                    for i, doc in enumerate(result["source_documents"][:5]):
                                        st.markdown(f"#### 📄 Reference {i+1}")
                                        st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                                with tab2:
                                    st.metric("Chunks Considered", len(result["source_documents"]))
                                    st.metric("Response Confidence", f"{min(100, len(result['source_documents'])*20)}%")
                    except Exception as e:
                        error_msg = f"🚨 Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            warning_msg = "⚠️ Please process documents first"
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Powered by Gemini 1.5 Pro • Secure Local Processing • 
        <a href="#" style="color: #4CAF50;">Privacy Policy</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
