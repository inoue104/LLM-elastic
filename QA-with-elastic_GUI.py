import os
import time
from os.path import join, dirname
from dotenv import load_dotenv
load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST")
ES_USER = os.environ.get("ES_USER")
ES_PASS = os.environ.get("ES_PASS")
ES_CERTS = os.environ.get("ES_CERTS")
INDEX_NAME = os.environ.get("INDEX_NAME")

import gradio as gr

import langchain
langchain.verbose = True
langchain.debug = True
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.schema.output_parser import StrOutputParser

def mail_qa(question):
    embeddings = OpenAIEmbeddings()
    vector_store = ElasticVectorSearch(
        ELASTICSEARCH_HOST,
        ssl_verify = {
                "basic_auth": (ES_USER, ES_PASS),
                "ca_certs": ES_CERTS,
            },
        index_name = INDEX_NAME,
        embedding = embeddings
    )

    # OpenAIへChat形式で問い合わせ
    llm = ChatOpenAI()

    # プロンプトの設定
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class expert on Apple products. You are answering questions about Apple products. Please answer the following question."), 
        ("user", "{input}")
    ])

    # ドキュメントプロンプトの設定
    prompt1 = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt1)

    # ドキュメント取得の設定
    retriever = vector_store.as_retriever()

    # 会話ができるように
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    chat_history = [HumanMessage(content="icloudの量を節約したいな"), AIMessage(content="はい!")]
    retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "どういう方法がありますか？日本語で回答してください。"
    })

    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt1)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = [HumanMessage(content="icloudの量を節約したいな"), AIMessage(content="はい!")]
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    res = response["answer"]
    return res

def slow_echo(message, history):
    answer = mail_qa(message)
    for i in range(len(answer)):
        answer1 = str(answer)
        time.sleep(0.1)
        yield "You said: " + answer1[: i+1]

gr.ChatInterface(slow_echo).launch()
#print(mail_qa("icloudの量を節約するためにはどういう方法がありますか？日本語で回答してください。"))




