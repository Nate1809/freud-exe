# app/memory_engine/utils.py

from typing import Union, List, Dict, Any

from langchain_core.messages import BaseMessage

def extract_text(user_input):
    if isinstance(user_input, BaseMessage):
        return user_input.text()
    elif isinstance(user_input, dict):
        return user_input.get('text') or user_input.get('content') or str(user_input)
    elif isinstance(user_input, list):
        texts = [extract_text(item) for item in user_input]
        return ' '.join(texts)
    elif isinstance(user_input, str):
        return user_input
    else:
        return str(user_input)
