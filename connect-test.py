from elasticsearch import Elasticsearch


host = "https://localhost:9200"
USER = "elastic"
PASS = "7_Mmrpl3M*-sL85dHhM5"
CERTIFICATE = "./http_ca.crt"

client = Elasticsearch(
    host, basic_auth=(USER, PASS), ca_certs=CERTIFICATE
)

print(client.info())

