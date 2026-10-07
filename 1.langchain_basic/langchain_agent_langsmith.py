import sys
from pathlib import Path

# 상위 폴더의 common_config.py를 import하기 위해 추가 (import보다 먼저 실행되어야 함)
# ".."은 실행 위치 기준이라 이 파일 위치 기준의 절대 경로를 사용
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common_config import get_llm
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


llm = get_llm("gpt-5.4-mini")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)
print(result)