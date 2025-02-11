import os

def create_upload_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created upload directory: {directory}")

from typing import List
import bs4

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_document_loaders(file_name):

    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8"),
    }
    
    for extension, loader in loaders.items():
        if file_name.endswith(extension):
            print(extension[1:], "loader를 return합니다.")
            return loader(file_name)
    
    return None

def get_web_loader(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
        },
    )
    return loader

def get_documents(loader, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split(text_splitter=text_splitter)

def get_vectorstore(doc_list):
    embeddings = OpenAIEmbeddings() 
    return faiss.FAISS.from_documents(doc_list, embeddings)

def get_prompt(usage):
    basic_prompt = PromptTemplate(
            template="""You are a helpful assistant. 
            Answer the following question. 
            Answer in Korean: {user_input}""",
            input_variables=["user_input"]
        )
    rag_prompt = hub.pull("rlm/rag-prompt")

    prompts = {
        "general": basic_prompt,
        "document_search": rag_prompt,
    }
    
    return prompts[usage]


def get_chain(usage, retriever):
    prompt = get_prompt(usage)
    if usage == "document_search":
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
    else:
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=1)
    return chain

