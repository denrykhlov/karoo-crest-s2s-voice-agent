"""Multilingual voice profiles and locale helpers for es-MX, es, and en."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_LANGUAGES: tuple[str, ...] = ("es-MX", "es", "en")


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Language-specific prompts, endpointing continuation words, and filler instructions."""

    code: str
    label: str
    short_label: str
    tts_accent: str
    call_pickup_phrase: str
    live_asr_system_prompt: str
    style_instructions: str
    filler_instruction: str
    router_language_name: str
    greeting_template: str
    continuation_words: frozenset[str]
    barge_in_exempt_words: frozenset[str]


_COMMON_EXEMPT = frozenset(
    {
        "stop",
        "wait",
        "hold",
        "hello",
        "excuse",
        "sorry",
        "espera",
        "espérame",
        "alto",
        "oye",
        "disculpa",
        "perdón",
        "perdon",
        "hola",
        "momento",
    }
)

_EN_CONTINUATION = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "as",
        "at",
        "because",
        "but",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "then",
        "to",
        "uh",
        "uhh",
        "um",
        "umm",
        "with",
        "your",
    }
)

_ES_CONTINUATION = frozenset(
    {
        "a",
        "al",
        "como",
        "con",
        "cuando",
        "de",
        "del",
        "donde",
        "e",
        "eh",
        "el",
        "emm",
        "en",
        "entre",
        "este",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mi",
        "ni",
        "o",
        "para",
        "pero",
        "por",
        "porque",
        "pues",
        "que",
        "qué",
        "si",
        "sin",
        "sobre",
        "su",
        "tu",
        "u",
        "un",
        "una",
        "unas",
        "unos",
        "y",
        "ya",
    }
)


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "es-MX": LanguageProfile(
        code="es-MX",
        label="Español (México)",
        short_label="es-MX",
        tts_accent="Spanish",
        call_pickup_phrase="¿Bueno?",
        live_asr_system_prompt="Perform live ASR.",
        style_instructions=(
            "Habla SIEMPRE en español natural y cordial de México (nunca uses 'vosotros', usa 'ustedes' o 'tú/usted'). "
            "Usa términos claros como 'cuota mensual', 'tasa de interés anual', 'sin comisión por cambio de divisa' y 'claro que sí'. "
            "Di los montos en rands sudafricanos de forma natural al hablar (por ejemplo, noventa y cinco rands al mes). "
            "Avanza la conversación en cada turno; nunca repitas presentaciones anteriores ni asumas respuestas, ingresos o datos que el cliente no haya dicho. "
            "Si el cliente responde sí o te pide continuar, comparte uno o dos beneficios clave y haz una sola pregunta sencilla. "
            "Responde en 2 oraciones cortas como máximo, sin listas, sin markdown y sin emojis."
        ),
        filler_instruction=(
            "[Solo di una frase breve y cordial de 3 a 6 palabras en español de México indicando que estás revisando la información, "
            "por ejemplo 'Claro que sí, déjame revisar eso.' Nada más.]"
        ),
        router_language_name="Mexican Spanish",
        greeting_template=(
            "El cliente acaba de contestar tu llamada saliente. Saluda brevemente y con calidez en español de México como "
            "{agent_name} de {company}, menciona la tarjeta de crédito {product} y pregunta amablemente si tiene un minuto rápido."
        ),
        continuation_words=_ES_CONTINUATION,
        barge_in_exempt_words=_COMMON_EXEMPT,
    ),
    "es": LanguageProfile(
        code="es",
        label="Español",
        short_label="es",
        tts_accent="Spanish",
        call_pickup_phrase="¿Sí, diga?",
        live_asr_system_prompt="Perform live ASR.",
        style_instructions=(
            "Habla SIEMPRE en español claro, cercano y profesional. "
            "Expresa las cifras de dinero en rands sudafricanos de forma natural al hablar (por ejemplo, noventa y cinco rands al mes). "
            "Avanza la conversación en cada turno; nunca repitas presentaciones anteriores ni asumas respuestas, ingresos o datos que el cliente no haya dicho. "
            "Si el cliente responde sí o te pide continuar, resume uno o dos beneficios clave y haz una pregunta breve para avanzar. "
            "Responde en 2 oraciones cortas como máximo, sin listas, sin markdown y sin emojis."
        ),
        filler_instruction=(
            "[Di únicamente una frase breve y amable de 3 a 6 palabras en español indicando que vas a consultar los detalles, "
            "por ejemplo 'Un momento, reviso los detalles.' Nada más.]"
        ),
        router_language_name="Spanish",
        greeting_template=(
            "El cliente acaba de contestar tu llamada saliente. Entrega un saludo cálido y breve en español presentándote como "
            "{agent_name} de {company}, mencionando la tarjeta {product} y preguntando con amabilidad si dispone de un minuto."
        ),
        continuation_words=_ES_CONTINUATION,
        barge_in_exempt_words=_COMMON_EXEMPT,
    ),
    "en": LanguageProfile(
        code="en",
        label="English",
        short_label="en",
        tts_accent="British English",
        call_pickup_phrase="Hello?",
        live_asr_system_prompt="Perform live ASR.",
        style_instructions=(
            "Speak ALWAYS in warm, polite South African / British English, voicing numbers naturally in Rand. "
            "Advance the conversation naturally every turn without repeating earlier introductions or availability checks, and wait for the caller to provide their own income or personal details before quoting numbers back. "
            "When the caller says yes, please continue, or go ahead, share one or two headline benefits and ask one simple discovery question. "
            "Keep every spoken reply concise (2 short sentences max), with zero markdown, bullet lists, or emojis."
        ),
        filler_instruction=(
            "[Say ONLY a brief 3-to-6 word warm holding phrase in English saying you are checking the details, "
            "for example 'Sure thing, let me check that.' Nothing else.]"
        ),
        router_language_name="English",
        greeting_template=(
            "The customer just answered your outbound call. Deliver a warm, brief opening greeting in English introducing "
            "yourself as {agent_name} from {company}, mentioning the {product}, and politely asking if they have a quick minute."
        ),
        continuation_words=_EN_CONTINUATION,
        barge_in_exempt_words=_COMMON_EXEMPT,
    ),
}


def normalize_language_code(raw: str | None) -> str:
    """Normalize user/config language input to one of `('es-MX', 'es', 'en')`."""
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("es-mx", "esmx", "mexican", "es-latam", "es-419"):
        return "es-MX"
    if cleaned.startswith("es"):
        return "es"
    if cleaned.startswith("en"):
        return "en"
    return "en"


def get_language_profile(code: str | None) -> LanguageProfile:
    """Return the LanguageProfile for a supported language code."""
    return LANGUAGE_PROFILES[normalize_language_code(code)]


def build_localized_system_prompt(
    base_prompt: str,
    lang_code: str,
    persona_name: str,
    voice_gender: str,
    delegation_mode: str = "selective_tool",
) -> str:
    """Adapt the base telesales system prompt with active language and delegation posture."""
    profile = get_language_profile(lang_code)
    if delegation_mode == "always_delegate":
        role_note = (
            "DELEGATION POSTURE (Always Delegate):\n"
            "- Your role is warm understanding, conversational turn-taking, and expressive speech delivery.\n"
            "- Complex banking rules, fees, rates, and policy details are retrieved from Karoo Crest's backend Knowledge Expert and passed into context.\n"
            "- Present retrieved facts faithfully without inventing fees or percentages outside provided context."
        )
    else:
        role_note = (
            "DELEGATION POSTURE (Selective Router + CRM Tools):\n"
            "- Handle conversational rapport directly, call local CRM tools when customer numbers/details are ready "
            "(pre-qualification, repayment calculation, POPIA-consented application, callback), "
            "and rely on retrieved Flametree Knowledge Expert facts whenever provided in context."
        )

    return (
        f"{base_prompt.strip()}\n\n"
        f"ACTIVE LANGUAGE & VOICE TARGET ({profile.code} — {profile.label}):\n"
        f"- You are ALWAYS {persona_name} (the advisor), speaking in a {voice_gender} voice ({profile.tts_accent}).\n"
        f"- {profile.style_instructions}\n\n"
        f"{role_note}"
    )


def build_localized_greeting_trigger(
    lang_code: str,
    persona_name: str,
    company_name: str = "Karoo Crest Bank",
    product_name: str = "Ubuntu Rewards Credit Card",
) -> str:
    """Build the greeting turn trigger in the active target language."""
    profile = get_language_profile(lang_code)
    return profile.greeting_template.format(
        agent_name=persona_name,
        company=company_name,
        product=product_name,
    )
