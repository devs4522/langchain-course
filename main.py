from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
        Ma, is a Chinese businessman and philanthropist. He is the founder of the Jack Ma Foundation, and co-founder of Alibaba Group and Yunfeng Capital. As of May 2025, Ma's net worth was estimated at US$27.2 billion.[2]

        After taking the gaokao three times, Ma earned a bachelor’s degree in English from Hangzhou Normal University in 1988 and was assigned as a lecturer at Hangzhou Dianzi University. A pioneer of Internet business in China, he founded a translation agency in 1994 and launched its website the following year, before leaving his university position to run an online yellow pages service, which was acquired by China Telecommunications Corporation in 1996. Following an uneasy collaboration with the state-owned enterprise, Ma left the company in 1997 and went on to develop websites for China’s Ministry of Foreign Trade and Economic Cooperation. In 1999, he co-founded Alibaba Group, initially as a business-to-business (B2B) e-commerce marketplace and later expanded into a multinational conglomerate.

        Ma has been the face of Chinese entrepreneurship. His influence declined after Chinese authorities halted the anticipated initial public offering (IPO) of his financial technology company, Ant Group, in 2020, following his public criticism of regulators for prioritizing risk aversion over innovation.
    """

    summary_template = """
        given the information {information} about a person, I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )   

    # llm = ChatOpenAI(model="gpt-5.4", temperature=0)
    llm = ChatOllama(model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
