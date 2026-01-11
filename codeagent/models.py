"""
Model configuration and client factory for Azure OpenAI models.

Note: GPT-5.2 works for basic chat but not with smolagents CodeAgent due to missing 'stop' parameter support. Use gpt-4.1 or gpt-4.1-mini for smolagents.

Usage:
    from models import get_model, chat, get_smolagent_model

    # Get a configured OpenAI client
    client, model_name, config = get_model("gpt-4.1")

    # Or use the convenience chat function
    response = chat("gpt-4.1", "What is the capital of France?")

    # Get a model for smolagents CodeAgent
    from smolagents import CodeAgent
    model = get_smolagent_model("gpt-4.1")
    agent = CodeAgent(tools=[], model=model)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from locagent root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Model configurations
MODEL_CONFIGS = {
    # GPT-4.1 (second Azure resource)
    "gpt-4.1": {
        "endpoint": "https://cielara-bkash02.openai.azure.com/openai/v1/",
        "deployment": "gpt-4.1",
        "api_key_env": "AZURE_OPENAI_API_KEY_SECOND",
        "use_max_completion_tokens": False,
        "smolagent_compatible": True,
    },
    # GPT-4.1 Mini (main Azure resource)
    "gpt-4.1-mini": {
        "endpoint": "https://cielara-research-resource.cognitiveservices.azure.com/openai/v1/",
        "deployment": "gpt-4.1-mini",
        "api_key_env": "AZURE_OPENAI_API_KEY_MAIN",
        "use_max_completion_tokens": False,
        "smolagent_compatible": True,
    },
    # GPT-5.2 (main Azure resource, uses max_completion_tokens)
    # Note: GPT-5.2 doesn't support 'stop' parameter, may have issues with smolagents
    "gpt-5.2": {
        "endpoint": "https://cielara-research-resource.cognitiveservices.azure.com/openai/v1/",
        "deployment": "gpt-5.2",
        "api_key_env": "AZURE_OPENAI_API_KEY_MAIN",
        "use_max_completion_tokens": True,
        "smolagent_compatible": False,  # 'stop' parameter not supported
    },
}

# Aliases for convenience
MODEL_ALIASES = {
    "gpt4": "gpt-4.1",
    "gpt4.1": "gpt-4.1",
    "gpt-4": "gpt-4.1",
    "4.1": "gpt-4.1",
    "gpt4-mini": "gpt-4.1-mini",
    "gpt4.1-mini": "gpt-4.1-mini",
    "mini": "gpt-4.1-mini",
    "4.1-mini": "gpt-4.1-mini",
    "gpt5": "gpt-5.2",
    "gpt5.2": "gpt-5.2",
    "gpt-5": "gpt-5.2",
    "5.2": "gpt-5.2",
}


def list_models() -> List[str]:
    """List all available model IDs."""
    return list(MODEL_CONFIGS.keys())


def get_model(model_id: str) -> Tuple[OpenAI, str, dict]:
    """
    Get an OpenAI client configured for the specified model.

    Args:
        model_id: Model identifier (e.g., "gpt-4.1-mini", "deepseek-v3")

    Returns:
        Tuple of (OpenAI client, deployment_name, config_dict)

    Raises:
        ValueError: If model_id is not recognized
        RuntimeError: If API key is not configured
    """
    # Resolve alias
    resolved_id = MODEL_ALIASES.get(model_id.lower(), model_id.lower())

    if resolved_id not in MODEL_CONFIGS:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")

    config = MODEL_CONFIGS[resolved_id]

    # Get API key
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set {config['api_key_env']} in .env")

    # Create client
    client = OpenAI(
        base_url=config["endpoint"],
        api_key=api_key,
    )

    return client, config["deployment"], config


def get_smolagent_model(model_id: str, **kwargs):
    """
    Get a model instance compatible with smolagents CodeAgent/ToolCallingAgent.

    Args:
        model_id: Model identifier (e.g., "gpt-4.1", "gpt-4.1-mini", "gpt-5.2")
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        A smolagents-compatible model instance (OpenAIServerModel)

    Example:
        from smolagents import CodeAgent
        from models import get_smolagent_model

        model = get_smolagent_model("gpt-4.1")
        agent = CodeAgent(tools=[], model=model)
        result = agent.run("What is 2+2?")
    """
    from smolagents import OpenAIServerModel

    # Resolve alias
    resolved_id = MODEL_ALIASES.get(model_id.lower(), model_id.lower())

    if resolved_id not in MODEL_CONFIGS:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")

    config = MODEL_CONFIGS[resolved_id]

    # Get API key
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"API key not found. Set {config['api_key_env']} in .env")

    # Create smolagents model using OpenAIServerModel (works with OpenAI-compatible APIs)
    model = OpenAIServerModel(
        model_id=config["deployment"],
        api_base=config["endpoint"],
        api_key=api_key,
        **kwargs
    )

    return model


def chat(
    model_id: str,
    message: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Convenience function for simple chat completions.

    Args:
        model_id: Model identifier
        message: User message
        system_prompt: Optional system prompt
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments passed to chat.completions.create

    Returns:
        Assistant's response text
    """
    client, deployment, config = get_model(model_id)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    params = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
        **kwargs
    }
    if max_tokens:
        # Some models (like gpt-5.2) require max_completion_tokens instead of max_tokens
        if config.get("use_max_completion_tokens", False):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message.content


def chat_with_history(
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Chat completion with full message history.

    Args:
        model_id: Model identifier
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments

    Returns:
        Assistant's response text
    """
    client, deployment, config = get_model(model_id)

    params = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
        **kwargs
    }
    if max_tokens:
        if config.get("use_max_completion_tokens", False):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message.content


# Test function
def test_models():
    """Test all configured models with a simple prompt."""
    test_prompt = "Say 'Hello' in one word."

    print("Testing models...")
    print("=" * 50)

    for model_id in list_models():
        print(f"\nTesting {model_id}...")
        try:
            response = chat(model_id, test_prompt, max_tokens=10)
            print(f"  ✓ Response: {response.strip()}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Test complete.")


def test_smolagent():
    """Test smolagent model with CodeAgent."""
    from smolagents import CodeAgent

    print("Testing smolagent integration...")
    print("=" * 50)

    for model_id in list_models():
        print(f"\nTesting {model_id} with CodeAgent...")
        try:
            model = get_smolagent_model(model_id)
            agent = CodeAgent(tools=[], model=model, max_steps=2)
            result = agent.run("What is 2+2? Just give the number.")
            print(f"  ✓ Response: {result}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Smolagent test complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_models()
        elif sys.argv[1] == "--test-smolagent":
            test_smolagent()
        elif sys.argv[1] == "--test-all":
            test_models()
            print("\n")
            test_smolagent()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python models.py [--test|--test-smolagent|--test-all]")
    else:
        # Quick test with default model
        print("Available models:", list_models())
        print("\nQuick test with gpt-4.1-mini...")
        try:
            response = chat("gpt-4.1-mini", "What is 2+2? Answer with just the number.")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
