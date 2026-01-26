import logging
import google.generativeai as genai

from config import IterRetGenConfig

# --- Component 2: Generator (Gemini) ---
class LLMGenerator:
    def __init__(self, config: IterRetGenConfig, api_key: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing LLM Generator with model: {config.llm_model_name}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.llm_model_name)
        self.config = config
        self.logger.info("LLM Generator initialized successfully")

    def generate(self, prompt: str) -> str:
        # Paper uses greedy decoding (temp=0) [cite: 106]
        self.logger.debug(f"Generating response with temperature={self.config.temperature}")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")
        
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=4096
        )
        response = self.model.generate_content(prompt, generation_config=generation_config)
        generated_text = response.text.strip()
        self.logger.debug(f"Generated response length: {len(generated_text)} characters")
        return generated_text
