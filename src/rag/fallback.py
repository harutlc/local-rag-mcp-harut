import asyncio
import ollama
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_HOST,
    FALLBACK_MODEL,
    FALLBACK_NUM_SUGGESTIONS,
    FALLBACK_TEMPERATURE,
    EXPANSION_TIMEOUT,
)


class FallbackResponse(BaseModel):
    """Exactly the object shape requested from Ollama via the `format` schema.

    Both fields are required: Ollama builds its decoding grammar from this
    schema, and optional fields let the model skip one entirely.
    """
    message: str
    suggestions: list[str]


class FallbackMessage(BaseModel):
    """What the fallback node hands back to the caller."""
    query: str
    message: str
    suggestions: list[str] = Field(default_factory=list)
    generated: bool = True          # False when the static message was used

    def render(self) -> str:
        """Format the message for display to the user."""
        text = self.message.strip()
        if self.suggestions:
            bullets = "\n".join(f"  • {s}" for s in self.suggestions)
            text += f"\n\nTry asking:\n{bullets}"
        return text


STATIC_MESSAGE = (
    "I could not work out how to search the knowledge base for that question. "
    "Try rewriting it with more specific wording - name the topic, document, or "
    "behaviour you are asking about."
)

SYSTEM_PROMPT = (
    "You help people search a company knowledge base of internal policies, "
    "procedures and technical documentation. A question has just failed to be "
    "processed into search terms. Your job is to tell the person what went "
    "wrong in plain language and show them how to ask it better. Never attempt "
    "to answer the question itself - you have not searched anything. "
    "Respond with JSON only."
)


def _build_user_prompt(query: str, reason: str | None) -> str:
    reason_line = f"\nInternal reason for the failure: {reason}\n" if reason else ""

    return f"""The person asked:
{query}
{reason_line}
Generate:
- message: one or two sentences, addressed to the person, saying that their
  question could not be turned into a search and briefly why it was hard to
  interpret. Refer to what they actually asked. Do not apologise more than
  once, do not mention JSON, models, or internal errors. End after those
  sentences: the rewritten questions do NOT belong here, and the message must
  not introduce or list them.
- suggestions: {FALLBACK_NUM_SUGGESTIONS} concrete rewrites of their question
  that would search well - specific, self-contained questions about the same
  topic. Write them as the person would type them. These are shown to the
  person separately, under their own heading."""


def _static(query: str) -> FallbackMessage:
    return FallbackMessage(query=query, message=STATIC_MESSAGE, generated=False)


def _strip_inlined_suggestions(message: str, suggestions: list[str]) -> str:
    """Cut the message where it starts repeating the suggestions.

    Small models inline the rewrites into the message however firmly the prompt
    says not to, and render() lists them again underneath. Rather than rely on
    the instruction holding, cut the message at the first suggestion it repeats
    and drop the dangling lead-in ("Here are some better ways to ask: 1)").
    """
    lowered = message.lower()
    cut = len(message)

    for suggestion in suggestions:
        probe = suggestion.lower()[:30].strip()
        if len(probe) < 10:
            continue
        found = lowered.find(probe)
        if found != -1:
            cut = min(cut, found)

    if cut == len(message):
        return message

    head = message[:cut]
    sentence_end = max(head.rfind("."), head.rfind("?"), head.rfind("!"))
    if sentence_end != -1:
        head = head[:sentence_end + 1]

    return head.strip()


async def generate_fallback(query: str, reason: str | None = None) -> FallbackMessage:
    """Ask the LLM for a message explaining that the query could not be searched.

    Used when query expansion produced nothing usable. Never raises: if this
    call fails too - which it will when Ollama itself is the problem - a fixed
    message is returned instead.
    """
    try:
        client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=EXPANSION_TIMEOUT)
        response = await client.chat(
            model=FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, reason)},
            ],
            format=FallbackResponse.model_json_schema(),
            options={"temperature": FALLBACK_TEMPERATURE},
        )
        parsed = FallbackResponse.model_validate_json(response["message"]["content"])

        message = parsed.message.strip()
        suggestions = []
        seen = set()
        for s in parsed.suggestions:
            cleaned = " ".join(str(s).replace("_", " ").split())
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                suggestions.append(cleaned)
            if len(suggestions) == FALLBACK_NUM_SUGGESTIONS:
                break

        message = _strip_inlined_suggestions(message, suggestions)
        if not message:
            return _static(query)

        return FallbackMessage(query=query, message=message, suggestions=suggestions)
    except Exception as e:
        print(f"⚠️  Warning: fallback message generation failed ({e}).")
        return _static(query)


def generate_fallback_sync(query: str, reason: str | None = None) -> FallbackMessage:
    """Blocking wrapper around generate_fallback for synchronous callers."""
    return asyncio.run(generate_fallback(query, reason))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "tell me about the thing with the bot"
    result = generate_fallback_sync(q, reason="expansion returned no keywords")

    print(f"\n❓ Query: {q}")
    print(f"   generated by LLM: {result.generated}\n")
    print(result.render())
