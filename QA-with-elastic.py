import os
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


#from langchain_community.vectorstores import ElasticsearchStore
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
#from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
#from langchain.schema.runnable import RunnablePassthrough
#from langchain.prompts import ChatPromptTemplate
#from langchain.schema.output_parser import StrOutputParser
import langchain
langchain.verbose = True
langchain.debug = True

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

'''
retriever = vector_store.as_retriever()

llm = OpenAI()

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 
    
    context: {context}
    Question: "{question}"
    Answer:
    """
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)

ans1 = chain.invoke("icloudの量を節約するためにはどういう方法がありますか？日本語で回答してください。")

print("---- Answer1 ----")
print(ans1)

ans2 = chain.invoke("大画面で写真を写したいのですが、どういう方法がありますか？日本語で回答してください。")

print("---- Answer2 ----")
print(ans2)
'''

# OpenAIへChat形式で問い合わせ
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

# プロンプトの設定
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class expert on Apple products. You are answering questions about Apple products. Please answer the following question."), 
    ("user", "{input}")
])

# ドキュメントプロンプトの設定
from langchain.chains.combine_documents import create_stuff_documents_chain
prompt1 = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt1)

# ドキュメント取得の設定
from langchain.chains import create_retrieval_chain
retriever = vector_store.as_retriever()

# 会話ができるように
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="icloudの量を節約したいな"), AIMessage(content="はい!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "どういう方法がありますか？日本語で回答してください。"
})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="icloudの量を節約したいな"), AIMessage(content="はい!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "どういう方法がありますか？日本語で回答してください。"
})

# ベクターストアからのドキュメント取得
#response = retriever_chain.invoke({"input": "icloudの量を節約するためにはどういう方法がありますか？日本語で回答してください。"})
print(response["answer"])

'''
# LLMの出力を整形
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# LLM chain の設定
chain = prompt | llm | output_parser

# LLM chainを使った問い合わせ
answer = chain.invoke({"input": "icloudの量を節約するためにはどういう方法がありますか？日本語で回答してください。"})
print(answer)
'''

