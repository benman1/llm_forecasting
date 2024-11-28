"""LLamaCPP integration.

We can use a 7B model bitnet model with 100k context window from here:
https://huggingface.co/gate369/Bitnet-M7-70m-Q8_0-GGUF
"""
from typing import List

from llama_cpp import Llama

class LlamaCppModel():

    def __init__(
            self,
            model_path: str,
            max_tokens: int = 256,
            temperature: float = 0.1,
            top_p: float = 0.5,
            echo: bool = False,
            stop: List[str] = ["#"],
        ):
        self.stop = stop
        self.echo = echo
        self.top_p = top_p
        self.temperature = temperature
        self.llm = self.load_llm(model_path=model_path)
        self.model_path = model_path
        self.max_tokens = max_tokens


    @staticmethod
    def load_llm(model_path: str, context_size: int = 512, batch_size: int = 126) -> Llama:
        """Load a LlamaCpp model."""
        return Llama(model_path, n_ctx=context_size, n_batch=batch_size)


    def generate_text(
            self,
            prompt: str,

        ) -> str:
        """Generate a text."""
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            echo=self.echo,
            stop=self.stop,
        )
        output_text = output["choices"][0]["text"].strip()
        return output_text


if __name__ == "__main__":
    llm_model = LlamaCppModel("/Users/ben/Downloads/bitnet-m7-70m.Q8_0.gguf")
    print(llm_model.generate_text("Hello!"))
