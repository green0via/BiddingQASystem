from pymongo import MongoClient

mongo = MongoClient("mongodb://localhost:27017/")
db = mongo["xunfei"]
col_mongo = db["中心制度规则_split"]

from pymilvus import MilvusClient, connections, utility, FieldSchema, CollectionSchema, DataType, Collection
connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="para_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
]

schema = CollectionSchema(fields, description="Policy Paragraph Embeddings")

col_name = "CentrePolicy"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

dense_index = {"index_type": "AUTOINDEX", "metric_type": "L2"}
col.create_index("dense_vector", dense_index)
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)

col.load()

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',
                      use_fp16=True,
                      pooling_method='cls',
                      devices=['cuda:0'])

def get_embeddings(text):
    embeddings = model.encode(
        text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    return embeddings

from tqdm import tqdm

batch_data = []
batch_ids = []
BATCH_SIZE = 100

for doc in tqdm(col_mongo.find({"vectorized": False}), desc='向量生成中...'):
    para_id = doc['para_id']
    text = doc['text']
    try:
        vector = get_embeddings(text)
        # print(vector)
        # break
        dense_vector = vector["dense_vecs"]
        sparse_vector = vector["lexical_weights"]
        batch_data.append([para_id, text, dense_vector, sparse_vector])
        batch_ids.append(doc['_id'])
        if len(batch_data) > BATCH_SIZE:
            col.insert(batch_data)
            col_mongo.update_many({"_id": {"$in": batch_ids}}, {"$set": {"vectorized": True}})
            batch_data = []
            batch_ids = []

    except Exception as e:
        print(f"向量化失败：{para_id}, {type(e).__name__}: {e}")

if batch_data:
    col.insert(batch_data)
    col_mongo.update_many({"_id": {"$in": batch_ids}}, {"$set": {"vectorized": True}})