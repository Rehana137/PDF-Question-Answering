from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Load PDF
docs = PyPDFLoader("dataset/sample.pdf").load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print("Pages:", len(docs), "| Chunks:", len(chunks))
print(chunks[0].page_content[:200] if chunks else "NO TEXT FOUND")

# 3. Embed and store
db = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"))
retriever = db.as_retriever(search_kwargs={"k": 3})

# 4. LLM + prompt
llm = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below. If the answer isn't there, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
chain = prompt | llm

# 5. Chat loop
print("Ask questions about the PDF (type 'exit' to quit)")
while True:
    q = input("\nYou: ")
    if q.lower() in ("exit", "quit"):
        break
    context = "\n\n".join(d.page_content for d in retriever.invoke(q))
    print("Bot:", chain.invoke({"context": context, "question": q}))