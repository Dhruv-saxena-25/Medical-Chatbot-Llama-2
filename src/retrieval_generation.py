from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA 
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from src.data_ingestion import data_ingestion


def generation(vstore):

    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferWindowMemory(k=6)

    MEDICAL_BOT_TEMPLATE="""
                    You are a medical chatbot Use the following pieces of information to answer the user's question.
                    If you don't know the answer use your own knowledge to give correcect answer. if the question  is 
                     out of the context just say that you don't know, don't try to make up an answer.
                    If person greats you great them back. please do not mention page numbers in answers.
                    
                    Context: 
                    {context}
                    
                    Question: 
                    {question}
                    
                    YOUR ANSWER:
                            
                    """
    

    prompt = PromptTemplate(template= MEDICAL_BOT_TEMPLATE, input_variables= ["context", "question"])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)

    print(llm)
   

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type= "stuff",
                                        retriever= retriever,
                                        input_key= "query",
                                        memory= memory,
                                        chain_type_kwargs= {"prompt": prompt})
    return chain

if __name__=='__main__':
    vstore = data_ingestion("done")
    chain  = generation(vstore)
    print(chain)
    
    result= chain("What is allergy?")
    print(result['result'])
    