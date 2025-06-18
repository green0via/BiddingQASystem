import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"
from models import glm_model, gemini_model
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from databases import AllProducts, AllCompanies, AllBiddings, AllPolicies
from databases import SQLDB
from Prompts import SQL_PROMPT

db = SQLDB

def recognise_intent(question):
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    prompt = f"""
    你是一个用户意图识别专家，擅长分析用户对话，并识别出用户的意图。作为一个智能问答系统的关键组件，你的工作是：

    1. 分析用户对话，识别出用户的意图。
    2. 根据用户的意图，准确输出对应的意图标签。
    3. 通常来说，问题中带有“项目”是指招标项目，但并不绝对，你需要自己判断。
    4. 在分析意图时，请一步一步分析。
    问答系统支持以下意图：

    - 查询政策信息
    - 查询招标信息
    - 查询商品信息
    - 查询企业信息

    ## 限制

    - 只输出意图标签，不要输出任何解释。
    - 不要输出意图标签之外的任何内容。
    - 如果无法识别出意图，请输出 `unknown`。
    """
    systemPrompts = [
        ("system", prompt),
    ]

    samples = [
        ("user", "香蕉可以空腹吃吗？"),
        ("assistant", "unknown"),

        ("user", "最新的三条政府采购公告"),
        ("assistant", "查询招标信息"),

        ("user", "招标失败怎么办"),
        ("assistant", "查询政策信息"),

        ("user", "静安区音乐节项目的负责人是谁？"),
        ("assistant", "查询招标信息"),

        ("user", "这个公司的法人是谁"),
        ("assistant", "查询企业信息"),

        ("user", "这个公司还代理了其他什么项目？"),
        ("assistant", "查询招标信息"),
        ("developer", "用户提到这个公司还代理了其他什么项目，说明该公司现在是现在这个项目的代理机构，助手应该查询其他项目中，代理机构是该公司的项目。")
        # 还可以在这继续将样本
        # 大模型会从样本中总结出规律，并应用到新对话中
    ]

    response = glm_model.invoke(systemPrompts + samples + [("user", question)]).content
    print(f"意图：{response.strip()}")
    return response.strip()


def if_sql(question):
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    prompt = """
    你是一个数据库管理机器人，我需要你根据用户提出的问题，判断是否需要检索SQL数据库。你的工作是：

    1、分析用户问题，判断是否是语义问题，或者是逻辑/条件问题。
    2、准确输出判断结果：
        - 若需要使用SQL语句检索数据库，输出"True"
        - 若**不**需要使用SQL数据库，输出"False"
    3、在判断是请一步一步进行分析。

    通常来说，问题只要涉及多条数据（搜索整个表）可能需要sql检索，如果只是针对某一条数据（搜索一条数据）不需要查询SQL数据库。
    仅供参考，具体以实际问题为主。

    ## 限制

    - 只输出判断结果，不要输出任何解释
    - 不要输出结果之外的任何内容，确保输出内容可以直接作为Bool值在python语句中运行
    """

    systemPrompts = [
        ("system", prompt),
    ]
    samples = [
        ("user", "告诉我最新的三条政府采购公告"),
        ("assistant", "True"),

        ("user", "上海静安音乐节项目的具体信息"),
        ("assistant", "False"),

        ("user", "上海市政府采购中心代理了哪些项目？"),
        ("assistant", "True"),

        ("user", "有关教育的政府采购公告有哪些？"),
        ("assistant", "True"),

        ("user", "这个公司还代理了其他什么项目？"),
        ("assistant", "True")
        # 还可以在这继续将样本
    ]

    response = glm_model.invoke(systemPrompts + samples + [("user", question)]).content
    print(f'是否需要SQL: {response.strip()}')
    return response.strip()

def str2bool(s):
    if s == 'False':
        return False
    else:
        return bool(s)


def create_sql_query(question, history, k=10):
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"
    sql_chain = create_sql_query_chain(gemini_model, db, prompt=SQL_PROMPT, k=k)
    response = sql_chain.invoke(
        {
            "question": question,
            "history": history
        }
    )
    return response


def validate_sql_query(query):
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"

    system = """Double check the user's {dialect} query for common mistakes, including:
    - Only return SQL Query not anything else like ```sql ... ```
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates\
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query.
    If there are no mistakes, just reproduce the original query with no further commentary.

    Output the final SQL query only."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect)

    validation_chain = prompt | gemini_model | StrOutputParser()
    query = validation_chain.invoke({"query": query})
    query = query.replace('```sql', '').replace('```', '')
    print(f'SQL Query: {query}')
    return query


def recognise(question, history, ifsql=False):
    if ifsql:
        query = validate_sql_query(create_sql_query(question=question, history=history))
    else:
        query = question
    return query


def choose_collection(intent):
    if '政策' in intent:
        col = AllPolicies
    elif '招标' in intent:
        col = AllBiddings
    elif '商品' in intent:
        col = AllProducts
    elif '企业' in intent:
        col = AllCompanies
    return col

def format_sql_docs(docs):
    return '\n'.join(doc[0] for doc in docs)

def run_sql(query):
    print(db.run(query))
    return db.run(query)