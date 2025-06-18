import IntentRecognition as ir
# from models import qwen_llm
from models import gemini_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from RAG import format_docs, hybrid_search, rerank
from functools import partial
from Prompts import GENERATOR_PROMPT_TEMPLATE

def main():
    question = input('请输入问题：')
    intent = ir.recognise_intent(question)
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
            query = ir.recognise(question, intent, ifsql=ifsql)
            context_chain = (
                    RunnableLambda(ir.run_sql)
                    | RunnableLambda(ir.format_sql_docs)
            )
        prompt = PromptTemplate(
            template=GENERATOR_PROMPT_TEMPLATE, input_variables=["context", "question", "history"]
        )
        # context_chain = RunnableLambda(hybrid_search_args) | RunnableLambda(rerank) | RunnableLambda(format_docs)
        # llm = RunnableLambda(qwen_llm)
        llm = gemini_model
        rag_chain = (
            {
                "context": context_chain,
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    else:
        query = question
        prompt = PromptTemplate(
            template=GENERATOR_PROMPT_TEMPLATE, input_variables=["history", "context", "question"]
        )
        # context_chain = RunnableLambda(hybrid_search_args) | RunnableLambda(rerank) | RunnableLambda(format_docs)
        # llm = RunnableLambda(qwen_llm)
        llm = gemini_model

        rag_chain = (
            {
                "context": RunnableLambda(lambda _: "无相关文档"),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    for s in rag_chain.stream(query):
        print(s, end="", flush=True)

if __name__ == '__main__':
    main()