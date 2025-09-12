from langchain.callbacks.base import BaseCallbackHandler

class PrintCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)

    def on_agent_action(self, action, **kwargs):
        print(f"\n[Agent Action]: {action}\n")

    def on_tool_end(self, output, **kwargs):
        print(f"\n[Tool Result]: {output}\n")

    def on_chain_end(self, outputs, **kwargs):
        print(f"\n[Chain End]: {outputs}\n")

    def get_output(self) -> str:
        return "".join(self.tokens)