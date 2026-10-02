import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

BASE_URL = "https://monogpt.kr/api/monorouter/v1"


def get_llm(
    model: str,
    temperature: float = 0,
    max_tokens: int = 2086,
    api_key: str | None = None,
):
    return ChatOpenAI(
        model=model,
        api_key=api_key or os.getenv("LLM_API_KEY"),
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )



def llm_connect(
    api_key: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 512,
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )


from langchain_openai import OpenAIEmbeddings

def embedding_model(model: str = "text-embedding-3-small"):
    return OpenAIEmbeddings(
        model=model,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=BASE_URL,
    )
