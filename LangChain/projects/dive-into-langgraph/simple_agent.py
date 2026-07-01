from langchain.agents import create_agent

from dive_config import create_chat_model

# 配置大模型服务
llm = create_chat_model()

# 创建Agent
agent = create_agent(model=llm)

# langgraph-cli 入口函数
def get_app():
    return agent
