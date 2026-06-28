"""
Lesson 01 - Model Comparison in LangChain
This example shows how to compare different AI models.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_models_to_compare() -> list[str]:
    raw = os.getenv("AI_COMPARE_MODELS", "")
    if raw.strip():
        models = [item.strip() for item in raw.split(",") if item.strip()]
        if models:
            return models
    current = os.getenv("AI_MODEL", "deepseek-v4-flash")
    return [current]


def compare_models():
    print("🔬 Comparing AI Models\n")

    prompt = "Explain recursion in programming in one sentence."
    models = get_models_to_compare()

    for model_name in models:
        print(f"\n📊 Testing: {model_name}")
        print("─" * 50)

        model = ChatOpenAI(
            model=model_name,
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        start_time = time.time()
        response = model.invoke(prompt)
        duration = (time.time() - start_time) * 1000

        print(f"Response: {response.content}")
        print(f"⏱️  Time: {duration:.0f}ms")

    print("\n✅ Comparison complete!")
    if len(models) >= 2:
        print("\n💡 Key Observations:")
        print("   - Compare quality, latency, and tool-calling stability")
        print("   - Prefer the cheaper/faster model for repetitive tasks")
        print("   - Prefer the stronger model for planning and long outputs")
    else:
        print("\n💡 Tip:")
        print("   - Set AI_COMPARE_MODELS in .env to compare multiple provider-specific models")


if __name__ == "__main__":
    compare_models()
