from pymilvus import connections, Collection
from pymongo import MongoClient
from langchain_community.utilities import SQLDatabase

SQLDB = SQLDatabase.from_uri("mysql+pymysql://readonly:1234@localhost:3306/xunfei")

connections.connect("default", host="localhost", port="19530")
AllPolicies = Collection("AllPolicies")
AllBiddings = Collection("AllBiddings")
AllProducts = Collection("AllProducts")
AllCompanies = Collection("AllCompanies")

mongo = MongoClient("mongodb://localhost:27017/")
gjzcfg = mongo["xunfei"]["国家按章分"]
shzchb = mongo["xunfei"]["上海按章分"]
zxzdgz = mongo["xunfei"]["中心按章分"]

def release():
    AllCompanies.release()
    AllPolicies.release()
    AllBiddings.release()
    AllProducts.release()
