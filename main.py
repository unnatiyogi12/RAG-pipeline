from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
data = PyPDFLoader("data/pdf/DSML.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarizes the text"),
     ("human", "{data}")]
)

chunks = splitter.split_documents(docs)

model = ChatMistralAI(model = "mistral-small-2506" \
"")

prompt = template.format_messages(data= docs)

result = model.invoke(prompt)
print(result.content)
