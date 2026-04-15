import getpass
import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your MistralAI API key: ")

docs = [
    Document(page_content="All about data visulaization understandingd of AI and ML model", metadata={"source":"AI/ML book"}),
    Document(page_content="pandas is uesed for data analysis", metadata={"source":"AI/ML book"})
]

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
)

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("What is used fro data analysis", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()
docs = retriver.invoke("explain deep learning")

for d in docs:
    print(d.page_content)

