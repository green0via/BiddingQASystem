from langchain_core.prompts import PromptTemplate


GENERATOR_PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If users ask logical question rather than contextual question, you 'd better provide the url of project to avoid potential mistake.
If the question is an inference question, you need to inferent step by step.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

<web>
{web}
<\web>

<history>
{history}
</history>

The response should be specific and use statistics or numbers when possible. 
If context is cannot recognise intent, let user ask more specific questions or answer base on your knowledge.
Do NOT say like "According to provided information...".
Never say "爬取时间" in your answer unless user ask your data updated date.
You can decide when to say these tips blow:
    - Note user you may make mistake, please double check important info.
    - If user asks a policy question, remind user that the above information may not be complete, and it is recommended to 
    consult the relevant laws and regulations for more detailed and accurate information.
    - At last you may remind user to get more info from provided url.
Answer in Chinese.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
generator_prompt = PromptTemplate(
    template=GENERATOR_PROMPT_TEMPLATE, input_variables=["history", "context", "question", "web"]
)

SQL_PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}
History: {history}
"""

SQL_DEFAULT_TEMPLATE = """Given an input question and chat history, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Note that when users ask the latest project or similar question, if you choose "爬取时间" column, you should order by "爬取时间"
ascending as data was scraped from top to end, so the latest project has the oldest time.
"""

SQL_PROMPT = PromptTemplate(
    input_variables=["input", "history", "table_info", "dialect", "top_k"],
    template=SQL_DEFAULT_TEMPLATE + SQL_PROMPT_SUFFIX,
)