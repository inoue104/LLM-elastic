# 同じフォルダにある、emails.csv をelasticsearch へ
# langchainとOpenAIを使い、csvファイルを読み込み、チャンクに分けて、
# OpenAIのEmbeddingsでベクター化した後、elasticへ
# kibanaの「Dev Tools」で「GET _search」にて確認

import os
from os.path import join, dirname
from dotenv import load_dotenv
load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST")
ES_USER = os.environ.get("ES_USER")
ES_PASS = os.environ.get("ES_PASS")
ES_CERTS = os.environ.get("ES_CERTS")
INDEX_NAME = os.environ.get("INDEX_NAME")

loader = CSVLoader(
    file_path="./emails.csv", 
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["From", "To", "Date", "Subject", "Message-Id", "In-Reply-To", "References", "Body"]
    },
)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(data)

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

def create_index(vector_store: ElasticVectorSearch):
    docs = splits
    vector_store.add_documents(
        docs,
        bulk_kwargs = {
            "chunk_size": 500,
            "max_chunk_bytes": 50000000,
        }
    )

if __name__ == "__main__":
    create_index(vector_store)



