import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain_community.document_loaders import GCSFileLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain_google_community import GCSDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro")

# Load, chunk and index the contents of the blog.
""" loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load() """

# Load, chunk and index the contents of a GCS directory.
# loader = GCSDirectoryLoader(project_name="basketball-data-analysis", bucket="ai_white_papers", continue_on_failure=True)
# docs = loader.load()

loader = UnstructuredPDFLoader("test_AI-Policy_LLM.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=VertexAIEmbeddings(model_name="textembedding-gecko"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What should be considered in the regulation of LLMs?"))