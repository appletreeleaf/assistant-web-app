from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_document_loaders(file_name):
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8")
    }
    
    for extension, loader in loaders.items():
        if file_name.endswith(extension):
            print(extension[1:], "loader를 return합니다.")
            return loader(file_name)
    
    return None

def get_documents(loader, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_split(text_splitter=text_splitter)

def get_vectorstore(doc_list):
    embeddings = OpenAIEmbeddings() 
    return faiss.FAISS.from_documents(doc_list, embeddings)

def get_chain(usage, retriever):
    if usage == "document_search":
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Answer in Korean.

            #Context: 
            {context}

            #Question:
            {question}

            #Answer:"""
        )
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
    else:
        prompt = PromptTemplate(
            template="""You are a helpful assistant. 
            Answer the following question. 
            Answer in Korean: {user_input}""",
            input_variables=["user_input"]
        )
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=1)
    return chain
