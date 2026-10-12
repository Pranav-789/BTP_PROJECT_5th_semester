import json
import logging
from typing import Dict, List, Optional, Tuple
from core.config import settings
from models.schemas import EvaluationResult, EvaluationStatus, ModelScore

logger = logging.getLogger(__name__)

BASIC_ERROR_STRINGS = [
    "error:",
    "traceback (most recent call last)",
    "exception:",
    "fatal:",
    "rate limit exceeded",
    "authentication failed",
    "invalid api key",
]

REFUSAL_TRIGGERS = [
    "i cannot assist with that",
    "i am unable to answer",
    "as an ai model, i cannot",
    "i apologize, but i cannot",
]


def is_response_acceptable(response: str) -> bool:
    """
    Phase 4 Validation Function:
    Checks for empty responses, whitespace-only content, or basic error strings.
    Returns True if acceptable, False otherwise.
    """
    if not response or not isinstance(response, str):
        return False

    cleaned = response.strip()
    if len(cleaned) < 10:
        return False

    lower_resp = cleaned.lower()
    for err in BASIC_ERROR_STRINGS:
        if err in lower_resp:
            logger.warning(f"Response rejected by basic validator: matched error string '{err}'")
            return False

    return True


class Agent2Evaluator:
    """
    Agent 2: Response Evaluator and Validator using LLM-as-a-Judge (Gemini 2.5 Flash)
    along with heuristic and basic error checks.
    """

    def __init__(self, max_retries: Optional[int] = None):
        self.max_retries = max_retries or settings.MAX_EVALUATOR_RETRIES
        self.validator_model = settings.VALIDATOR_MODEL  # gemini-2.5-flash
        self._validator_client = None
        self._init_validator_client()

    def _init_validator_client(self):
        gemini_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._validator_client = ChatGoogleGenerativeAI(
                    model=self.validator_model,
                    google_api_key=gemini_key,
                    temperature=0.1,
                    max_output_tokens=512
                )
                logger.info(f"Agent 2 initialized LLM validator with {self.validator_model}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini validator client: {e}")
                self._validator_client = None

    def validate_response(self, response_text: str, prompt: Optional[str] = None, selected_model: str = "llm") -> Tuple[bool, str]:
        """
        Validates response combining basic acceptance check and optional Gemini judge.
        """
        if not is_response_acceptable(response_text):
            return False, "Response is empty, too short, or contains error strings."

        lower_resp = response_text.lower()
        for trigger in REFUSAL_TRIGGERS:
            if trigger in lower_resp:
                return False, f"Model refused instruction: detected '{trigger}'."

        words = response_text.split()
        if len(words) > 30:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.20:
                return False, f"Degenerative repetition loop detected (unique ratio {unique_ratio:.2f})."

        # Optional LLM Judge if Gemini API key is configured and prompt is passed
        if self._validator_client is not None and prompt:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                system_prompt = (
                    "You are an impartial AI judge. Evaluate whether the response answers the prompt. "
                    'Respond strictly in JSON: {"valid": true, "reason": "..."}'
                )
                user_msg = f"User Prompt:\n{prompt}\n\nCandidate Response ({selected_model}):\n{response_text[:1500]}"
                judge_response = self._validator_client.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_msg)
                ])
                raw = judge_response.content if hasattr(judge_response, "content") else str(judge_response)
                if "{" in raw and "}" in raw:
                    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    return bool(data.get("valid", True)), f"Gemini 2.5 Flash: {data.get('reason', 'OK')}"
            except Exception as e:
                logger.warning(f"Gemini LLM validation skipped: {e}")

        return True, "Response accepted."

    def execute_with_guardrails(
        self,
        prompt: str,
        ranked_models: List[ModelScore],
        executor_func,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Tuple[str, str, EvaluationResult, Dict]:
        """
        Runs LLM execution pipeline with automated retry guardrails.
        """
        retries = 0
        total_candidates = len(ranked_models)

        for idx, candidate in enumerate(ranked_models):
            current_model = candidate.model_name
            logger.info(f"Agent 2 monitoring attempt {retries + 1} with model '{current_model}'")

            response_text, exec_meta = executor_func(
                model_name=current_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            is_valid, reason = self.validate_response(
                response_text=response_text,
                prompt=prompt,
                selected_model=current_model
            )

            if is_valid:
                eval_res = EvaluationResult(
                    status=EvaluationStatus.PASSED if retries == 0 else EvaluationStatus.REMEDIATED_FALLBACK,
                    is_valid=True,
                    evaluation_notes=reason,
                    retry_count=retries
                )
                return current_model, response_text, eval_res, exec_meta

            logger.warning(f"Agent 2 rejected output from '{current_model}': {reason}")
            retries += 1
            if retries > self.max_retries or (idx + 1) >= total_candidates:
                eval_res = EvaluationResult(
                    status=EvaluationStatus.VALIDATION_FAILED_BEST_EFFORT,
                    is_valid=False,
                    evaluation_notes=f"Exhausted {retries} retries. Last feedback: {reason}",
                    retry_count=retries
                )
                return current_model, response_text, eval_res, exec_meta

        # Fallback safeguard
        fallback_model = settings.FALLBACK_MODEL
        resp, meta = executor_func(fallback_model, prompt, temperature, max_tokens)
        eval_res = EvaluationResult(
            status=EvaluationStatus.PASSED,
            is_valid=True,
            evaluation_notes=f"Executed default fallback model ({fallback_model}).",
            retry_count=retries
        )
        return fallback_model, resp, eval_res, meta


def get_agent_2_evaluator() -> Agent2Evaluator:
    return Agent2Evaluator()
