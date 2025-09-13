from langchain.memory import ConversationBufferMemory
from typing import Any
import json

class SafeConversationMemory(ConversationBufferMemory):
    """Custom Conversation Buffer Memory for storing json"""
    def save_context(self, inputs : dict[str, Any], outputs : dict[str, Any]) -> None:
        safe_outputs : dict[str, str] = {}
        for k, v in outputs.items():
            if not isinstance(v, str):
                try:
                    safe_outputs[k] = json.dumps(v, indent = 2, ensure_ascii = False)
                except Exception:
                    safe_outputs[k] = str(v)
            else:
                safe_outputs[k] = v
        super().save_context(inputs, safe_outputs)