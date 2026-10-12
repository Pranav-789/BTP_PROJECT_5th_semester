import time
import logging
from typing import Dict, Optional, Tuple
from core.config import settings

logger = logging.getLogger(__name__)


class LLMExecutorService:
    """
    Executes routed LLM prompts using LangChain and provider SDKs.
    Supports Google Gemini (gemini-2.5-flash fallback/primary), OpenAI, Groq, and HuggingFace.
    Gracefully handles missing API keys with informative simulated output.
    """

    def __init__(self):
        self.fallback_model = settings.FALLBACK_MODEL

    def _get_langchain_client(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024):
        """
        Initializes appropriate LangChain Chat model instance based on available API keys.
        Maps GPT models to Gemini equivalents (gemini-2.5-flash-lite / gemini-2.5-flash) when OpenAI key is absent.
        """
        gemini_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not gemini_key and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("AQ."):
            gemini_key = settings.OPENAI_API_KEY

        normalized_name = model_name.lower()

        # 1. Google Gemini
        if "gemini" in normalized_name:
            if gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    return ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=gemini_key,
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    ), "Google Gemini"
                except Exception as e:
                    logger.warning(f"Failed to instantiate Gemini client: {e}")

        # 2. OpenAI Models (or Gemini Proxy when using Gemini key in place of OpenAI key)
        if "gpt" in normalized_name:
            # If an actual OpenAI key (sk-...) is configured, use OpenAI
            if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
                try:
                    from langchain_openai import ChatOpenAI
                    openai_model = "gpt-4o" if ("gpt_4" in normalized_name or "gpt-4" in normalized_name) else "gpt-3.5-turbo"
                    return ChatOpenAI(
                        model=openai_model,
                        api_key=settings.OPENAI_API_KEY,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ), "OpenAI"
                except Exception as e:
                    logger.warning(f"Failed to instantiate OpenAI client: {e}")

            # Proxy to active Gemini models:
            # gpt_3.5_turbo -> gemini-3.5-flash-lite
            # gpt_4 -> gemini-3.8-flash
            if gemini_key:
                is_gpt4 = "gpt_4" in normalized_name or "gpt-4" in normalized_name
                gemini_proxy_model = "gemini-3.8-flash" if is_gpt4 else "gemini-3.5-flash-lite"
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    return ChatGoogleGenerativeAI(
                        model=gemini_proxy_model,
                        google_api_key=gemini_key,
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    ), f"Google Gemini ({model_name} Proxy: {gemini_proxy_model})"
                except Exception as e:
                    logger.warning(f"Failed to instantiate Gemini proxy client ({gemini_proxy_model}): {e}")

        # 3. Open-source models via Groq
        if settings.GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                groq_model = "openai/gpt-oss-120b" if ("70b" in normalized_name or "65b" in normalized_name) else "openai/gpt-oss-20b"
                return ChatGroq(
                    model_name=groq_model,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=temperature,
                    max_tokens=max_tokens
                ), "Groq"
            except Exception as e:
                logger.warning(f"Failed to instantiate Groq client: {e}")

        # 4. Fallback to Gemini if Gemini Key is available
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.fallback_model,
                    google_api_key=gemini_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                ), "Google Gemini (Fallback)"
            except Exception as e:
                logger.warning(f"Failed to instantiate Gemini fallback: {e}")

        # 5. HuggingFace Endpoint if key exists
        if settings.HUGGINGFACE_API_KEY:
            try:
                from langchain_community.llms import HuggingFaceEndpoint
                hf_repo = "meta-llama/Llama-2-7b-chat-hf"
                return HuggingFaceEndpoint(
                    repo_id=hf_repo,
                    huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                ), "HuggingFace"
            except Exception as e:
                logger.warning(f"Failed to instantiate HuggingFace client: {e}")

        return None, "Simulation"

    def execute(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Tuple[str, Dict]:
        """
        Executes generation using LangChain pipeline or fallback simulation.
        Returns: (generated_text, execution_metadata)
        """
        start_time = time.time()
        client, provider = self._get_langchain_client(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        metadata = {
            "requested_model": model_name,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        gemini_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not gemini_key and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("AQ."):
            gemini_key = settings.OPENAI_API_KEY

        if client is not None and provider != "Simulation":
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=f"You are {model_name}, a helpful and truthful AI assistant."),
                    HumanMessage(content=prompt)
                ]
                logger.info(f"Invoking {provider} with model {model_name}...")
                response = client.invoke(messages)
                raw_content = response.content if hasattr(response, "content") else str(response)
                if isinstance(raw_content, list):
                    response_text = "\n".join(b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in raw_content)
                else:
                    response_text = str(raw_content)

                metadata["execution_time_sec"] = round(time.time() - start_time, 3)
                metadata["is_simulated"] = False
                return response_text, metadata

            except Exception as e:
                logger.error(f"Live LLM call to {provider} failed: {e}. Attempting Gemini fallback.")
                if gemini_key:
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        logger.info(f"Retrying live generation with Gemini fallback: {self.fallback_model}...")
                        fb_client = ChatGoogleGenerativeAI(
                            model=self.fallback_model,
                            google_api_key=gemini_key,
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                        response = fb_client.invoke(messages)
                        response_text = response.content if hasattr(response, "content") else str(response)
                        metadata["execution_time_sec"] = round(time.time() - start_time, 3)
                        metadata["is_simulated"] = False
                        metadata["provider"] = f"Google Gemini (Fallback: {self.fallback_model})"
                        return response_text, metadata
                    except Exception as fb_err:
                        logger.error(f"Gemini fallback retry failed: {fb_err}")

                metadata["error"] = str(e)

        # Simulation mode when API keys are not filled yet
        metadata["execution_time_sec"] = round(time.time() - start_time, 3)
        metadata["is_simulated"] = True

        simulated_response = (
            f"[{model_name} Response (Fallback: {self.fallback_model})]:\n\n"
            f"Here is the generated response for your prompt:\n\"{prompt}\"\n\n"
            f"Note: Generated in thesis test mode. Set GEMINI_API_KEY or GROQ_API_KEY in .env for live API generation."
        )

        return simulated_response, metadata


def get_llm_executor() -> LLMExecutorService:
    return LLMExecutorService()


async def execute_query(prompt: str, selected_model: str) -> str:
    """
    Phase 3: Async execution function using pattern routing (match/case)
    to dispatch to the appropriate provider SDK / API.
    Returns the generated text string.
    """
    executor = get_llm_executor()
    normalized = selected_model.lower()

    # Route using match / case statement based on model family
    match normalized:
        case _ if "gpt" in normalized:
            logger.info(f"Routing to OpenAI family executor: {selected_model}")
            text, _ = executor.execute(selected_model, prompt)
            return text

        case _ if "gemini" in normalized:
            logger.info(f"Routing to Google Gemini family executor: {selected_model}")
            text, _ = executor.execute(selected_model, prompt)
            return text

        case _ if any(k in normalized for k in ["llama", "falcon", "vicuna", "wizard", "mpt", "star"]):
            logger.info(f"Routing to open-source / Groq / HuggingFace executor: {selected_model}")
            text, _ = executor.execute(selected_model, prompt)
            return text

        case _:
            logger.info(f"Routing to generic / fallback executor: {selected_model}")
            text, _ = executor.execute(selected_model, prompt)
            return text
