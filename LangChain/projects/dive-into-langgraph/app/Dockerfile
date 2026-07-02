FROM python:3.13-slim

# 设置工作目录
WORKDIR /workspace

# 更换为国内镜像源并安装系统依赖
# 如需使用 npx 请安装 nodejs npm 
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    curl \
    supervisor \
    procps \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制根目录 uv 配置并安装 Python 依赖
COPY pyproject.toml uv.lock .python-version ./
COPY Agent/projects/memU/pyproject.toml Agent/projects/memU/
COPY RAG/projects/RAG-Techniques/pyproject.toml RAG/projects/RAG-Techniques/
RUN pip install uv==0.11.16 -i https://mirrors.aliyun.com/pypi/simple/
RUN uv sync --locked --no-dev --group langgraph-app -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY LangChain/projects/dive-into-langgraph/app/ ./LangChain/projects/dive-into-langgraph/app/
WORKDIR /workspace/LangChain/projects/dive-into-langgraph/app

# 复制supervisor配置文件
COPY LangChain/projects/dive-into-langgraph/app/docker.conf /etc/supervisor/conf.d/supervisord.conf

# 设置入口点
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
