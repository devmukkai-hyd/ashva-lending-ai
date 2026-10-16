"""
judges.py - M8: LLM-judge scores for the policy answers (used by `python evaluate.py --judges`).
evaluate.py calls run(name, ...) for each judge; a missing library or a failed call raises, and
evaluate.py skips just that score.

Three judges, the same ones as the course notebook. Each one asks a model to grade the answer:

  deepeval_correctness   DeepEval GEval   - does the answer say the same as the expected answer?
  ragas_faithfulness     Ragas            - is every claim in the answer supported by the retrieved policy text?
  trulens_groundedness   TruLens          - same question, scored by a different library

Scores are 0..1 (higher is better).

Judge model: EVAL_JUDGE_MODEL if set (e.g. "groq/qwen/qwen3-32b" with GROQ_API_KEY), otherwise your
LAB_MODEL. Using the same model family to write and to judge is weaker (a model tends to approve its own
style of answer) - a judge from another family is the better setup.

pip install deepeval ragas trulens trulens-providers-litellm   (+ langchain-litellm for a non-Gemini judge)
"""
import asyncio
import os
import re
import time

import risk_agent as ra

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or ra.MODEL

# ---------------------------------------------------------------------------
# Rate limit: free-tier Gemini allows 15 requests a MINUTE per model. One judge makes several model calls
# inside its library (TruLens: one per statement; Ragas: 2+; DeepEval: 1-2), so spacing the judges is not
# enough - EVERY model call is spaced instead:
#   * LiteLLM calls (the policy answers, DeepEval, TruLens) go through _paced(): >= MIN_GAP_SECONDS apart,
#     and on a 429 they wait as long as Google asks ("Please retry in 42s") and try again.
#   * Ragas uses Google's LangChain client, which gets LangChain's own rate limiter at the same pace.
# 4.5 s apart = ~13 calls a minute. A paid key: EVAL_MIN_GAP=0.
# ---------------------------------------------------------------------------
MIN_GAP_SECONDS = float(os.getenv("EVAL_MIN_GAP", "7.5"))
MAX_RATE_LIMIT_RETRIES = 3
_last_call = [0.0]


def _pace():
    wait = MIN_GAP_SECONDS - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _retry_delay(error) -> float | None:
    """Seconds the server asks us to wait, if this is a rate-limit error (None otherwise)."""
    text = str(error)
    if not ("429" in text or "RateLimit" in type(error).__name__ or "RESOURCE_EXHAUSTED" in text):
        return None
    if "PerDay" in text:
        return None                     # the DAILY quota: waiting a minute won't help
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", text, re.I) or re.search(r'retryDelay"?:\s*"(\d+)s', text)
    return float(m.group(1)) + 1 if m else 30.0


def _paced(fn):
    def call(*args, **kwargs):
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            _pace()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                delay = _retry_delay(e)
                if delay is None or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"   rate limited: waiting {delay:.0f}s as Google asks", flush=True)
                time.sleep(delay)
    return call


def _paced_async(fn):
    async def call(*args, **kwargs):
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            _pace()
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                delay = _retry_delay(e)
                if delay is None or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"   rate limited: waiting {delay:.0f}s as Google asks", flush=True)
                await asyncio.sleep(delay)
    return call


_originals: dict = {}


def install_pacer():
    """Route every LiteLLM call in this process through the pacer. Safe to call repeatedly: libraries that
    did `from litellm import completion` hold their own reference, so each call also swaps those."""
    import sys
    import litellm
    if not _originals:
        _originals["completion"], _originals["acompletion"] = litellm.completion, litellm.acompletion
        _originals["paced"] = {id(litellm.completion): _paced(litellm.completion),
                               id(litellm.acompletion): _paced_async(litellm.acompletion)}
        litellm.completion = _originals["paced"][id(_originals["completion"])]
        litellm.acompletion = _originals["paced"][id(_originals["acompletion"])]
    swaps = _originals["paced"]
    for module in list(sys.modules.values()):
        for name in ("completion", "acompletion"):
            ref = getattr(module, name, None) if module else None
            if ref is not None and id(ref) in swaps:
                try:
                    setattr(module, name, swaps[id(ref)])
                except Exception:
                    pass


def _rate_limiter():
    """LangChain's limiter for Ragas' Google client: same pace as the LiteLLM pacer."""
    if MIN_GAP_SECONDS <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(requests_per_second=1 / MIN_GAP_SECONDS, check_every_n_seconds=0.2, max_bucket_size=1)


def _with_retry(fn, tries: int = 2):
    """Run a judge. Rate limits are handled per call (above); this retries a judge once more if a library
    gave up anyway, after the wait Google asks for."""
    for attempt in range(tries):
        install_pacer()                              # also catches libraries imported since the last judge
        try:
            return fn()
        except ImportError:
            raise                                   # library not installed: retrying won't help
        except Exception as e:
            delay = _retry_delay(e)
            if delay is None or attempt == tries - 1:
                raise
            print(f"   judge rate limited: waiting {delay:.0f}s", flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# The three judges
# ---------------------------------------------------------------------------
def _deepeval_judge():
    """DeepEval's LiteLLM judge, minus log-probabilities.
    GEval first asks the model for token log-probabilities (top_logprobs) to weight its 1-10 score.
    Gemini does not offer them and LiteLLM refuses the request (UnsupportedParamsError). Raising
    AttributeError here is how DeepEval learns a custom model has no logprobs: it then asks for the
    score in plain JSON instead - the same score, without the probability weighting."""
    from deepeval.models import LiteLLMModel

    class NoLogprobsLiteLLM(LiteLLMModel):
        def generate_raw_response(self, *args, **kwargs):
            raise AttributeError("this judge model does not return logprobs")

        async def a_generate_raw_response(self, *args, **kwargs):
            raise AttributeError("this judge model does not return logprobs")

    return NoLogprobsLiteLLM(model=JUDGE_MODEL, api_key=None if os.getenv("EVAL_JUDGE_MODEL") else ra.GEMINI_API_KEY)


def deepeval_correctness(question, answer, expected):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    judge = _deepeval_judge()
    metric = GEval(name="Correctness",
                   criteria="Does the actual output state the same facts and figures as the expected output? "
                            "Missing or wrong figures lower the score; extra wording does not.",
                   evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                      LLMTestCaseParams.EXPECTED_OUTPUT],
                   model=judge, async_mode=False)
    metric.measure(LLMTestCase(input=question, actual_output=answer, expected_output=expected))
    return metric.score, metric.reason


def _langchain_judge_llm():
    """A LangChain chat model for Ragas.
    Gemini: langchain-google-genai (already installed - the policy index uses it for embeddings).
    Anything else (e.g. groq/...): ChatLiteLLM, which now lives in the langchain-litellm package
    (newer langchain-community removed it): pip install langchain-litellm"""
    provider, _, name = JUDGE_MODEL.partition("/")
    if (provider == "gemini" and name) or (not name and JUDGE_MODEL.startswith("gemini")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=name or JUDGE_MODEL, rate_limiter=_rate_limiter(),
                                      google_api_key=os.getenv("GEMINI_API_KEY") or ra.GEMINI_API_KEY)
    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError:
        from langchain_community.chat_models.litellm import ChatLiteLLM      # older langchain-community
    return ChatLiteLLM(model=JUDGE_MODEL, rate_limiter=_rate_limiter())


def _ragas_compat():
    """Older Ragas imports Vertex AI classes from langchain-community at load time (only to check model types):
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    Newer langchain-community removed them, so `import ragas` fails. We don't use Vertex AI, so give Ragas
    empty stand-ins. (Upgrading ragas - pip install -U ragas - is the other way out.)"""
    import sys
    import types

    class _Unused:                       # never instantiated; Ragas only does isinstance() checks with it
        pass

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = _Unused
        sys.modules["langchain_community.chat_models.vertexai"] = stub
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except ImportError:
        import langchain_community.llms as llms
        llms.VertexAI = _Unused


def ragas_faithfulness(question, answer, contexts):
    _ragas_compat()
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    llm = LangchainLLMWrapper(_langchain_judge_llm())
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
    return _run_async(lambda: Faithfulness(llm=llm).single_turn_ascore(sample)), ""


def _run_async(make_coro):
    """asyncio.run(), also when called inside Langfuse's run_experiment (which already runs an event
    loop, where asyncio.run is not allowed): then the coroutine runs in a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


def trulens_groundedness(answer, contexts):
    from trulens.providers.litellm import LiteLLM
    provider = LiteLLM(model_engine=JUDGE_MODEL)
    score, reasons = provider.groundedness_measure_with_cot_reasons(source="\n\n".join(contexts), statement=answer)
    return score, str(reasons)[:300]


def run(name: str, question: str, answer: str, contexts: list[str], expected: str):
    """Run one judge by name -> (score 0..1, reason). Raises if the library is missing or the call fails
    (evaluate.py's component_scorer catches that and skips just this one score)."""
    if JUDGE_MODEL == ra.MODEL and ra.GEMINI_API_KEY:
        os.environ.setdefault("GEMINI_API_KEY", ra.GEMINI_API_KEY)      # LiteLLM reads this name for Gemini
    jobs = {"deepeval_correctness": lambda: deepeval_correctness(question, answer, expected),
            "ragas_faithfulness": lambda: ragas_faithfulness(question, answer, contexts),
            "trulens_groundedness": lambda: trulens_groundedness(answer, contexts)}
    score, reason = _with_retry(jobs[name])
    return round(float(score), 2), reason or ""
