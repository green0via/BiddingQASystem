from models import bgem3_model, bge_rf
from pymilvus import AnnSearchRequest, WeightedRanker

def get_embeddings(text):
    embeddings = bgem3_model.encode(
        text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    return embeddings

def hybrid_search(
    col,
    # query_dense_embedding,
    # query_sparse_embedding,
    query,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=5,
):
    print("Embedding...")
    query_embeddings = get_embeddings([query])
    query_dense_embeddings = query_embeddings['dense_vecs'][0]
    query_sparse_embeddings = query_embeddings.get('lexical_weights')[0]

    print("Searching...")
    dense_req = AnnSearchRequest(
        [query_dense_embeddings], "dense_vector", {"metric_type": "L2", "params": {}}, limit=limit
    )
    sparse_req = AnnSearchRequest(
        [query_sparse_embeddings], "sparse_vector", {"metric_type": "IP", "params": {}}, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text"]
    )
    return [
        {"text": hit.entity.get("text")}
        for hit in res[0]
    ]

def rerank(results, query):
    print("Reranking...")
    documents = [p['text'] for p in results]
    rerank_results = bge_rf(query=query, documents=documents, top_k=5)
    return rerank_results

def format_docs(docs):
    return "\n\n".join(doc.text for doc in docs)

