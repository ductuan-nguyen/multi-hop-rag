import requests

url = "http://127.0.0.1:1610/v1/search"

def search(query):
    input = {'query': query, 'top_k': 5, 'top_k_rerank': 5, 'rerank': False, 'collection_name': 'web_mining'}
    resp = requests.post(url, json=input)
    output = []
    for item in resp.json()['results']:
        output.append({
            'doc_id': item['doc_id'],
            'content': item['text'],
        })
    return output