# app/memory_engine/utils.py

from typing import Union, List, Dict, Any

def extract_text(user_input: Union[str, List, Dict[str, Any]]) -> str:
    """Extract text content from various input formats."""
    if isinstance(user_input, list) and user_input:
        if isinstance(user_input[0], dict):
            if 'text' in user_input[0]:
                return user_input[0]['text']
            elif 'content' in user_input[0]:
                return user_input[0]['content']
    if isinstance(user_input, dict):
        if 'text' in user_input:
            return user_input['text']
        elif 'content' in user_input:
            return user_input['content']
    return str(user_input)
