# core.py
import IntentRecognition as ir
from models import gemini_temp
# from models import qwen_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from RAG import format_docs, hybrid_search, rerank
from functools import partial
from Prompts import GENERATOR_PROMPT_TEMPLATE
from tools import graph


# 封装成函数，供后端调用
def get_answer(question: str, history: str = "", use_search: bool = False) -> str:
    intent = ir.recognise_intent(question)
    if use_search:
        web_context = graph.invoke({"question": question})['answer']
    else:
        web_context = ''
    if intent != 'unknown':
        ifsql = ir.str2bool(ir.if_sql(question))
        if not ifsql:
            query = question
            col = ir.choose_collection(intent)
            col.load()

            hybrid_search_args = partial(
                hybrid_search,
                col,
                sparse_weight=1.0,
                dense_weight=1.0,
                limit=5
            )

            context_chain = (
                    RunnableLambda(hybrid_search_args)
                    | RunnableLambda(partial(rerank, query=query))
                    | RunnableLambda(format_docs)
            )
        else:
            query = ir.recognise(question, history, ifsql=ifsql)
            context_chain = (
                    RunnableLambda(ir.run_sql)
                    | RunnableLambda(ir.format_sql_docs)
            )
        prompt = PromptTemplate(
            template=GENERATOR_PROMPT_TEMPLATE, input_variables=["context", "question", "history", "web"]
        )
        # context_chain = RunnableLambda(hybrid_search_args) | RunnableLambda(rerank) | RunnableLambda(format_docs)
        # llm = RunnableLambda(qwen_llm)
        llm = gemini_temp
        rag_chain = (
                {
                    "context": context_chain,
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(lambda _: history),
                    "web": RunnableLambda(lambda _: web_context)
                }
                | prompt
                | llm
                | StrOutputParser()
        )

    else:
        query = question
        prompt = PromptTemplate(
            template=GENERATOR_PROMPT_TEMPLATE, input_variables=["context", "question", "history"]
        )
        # context_chain = RunnableLambda(hybrid_search_args) | RunnableLambda(rerank) | RunnableLambda(format_docs)
        # llm = RunnableLambda(qwen_llm)
        llm = gemini_temp


        rag_chain = (
                {
                    "context": RunnableLambda(lambda _: '无法识别意图'),
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(lambda _: history),
                    "web": RunnableLambda(lambda _: web_context)
                }
                | prompt
                | llm
                | StrOutputParser()
        )

    result = rag_chain.invoke(query)  # 可改为stream(query)支持流式
    # for s in rag_chain.stream(query):
    #     print(s, end="", flush=True)
    return result
