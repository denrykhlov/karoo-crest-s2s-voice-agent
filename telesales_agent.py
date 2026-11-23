#!/usr/bin/env python3
"""Karoo Crest Bank — South African Credit Card Telesales AI Voice Agent.

Powered end-to-end by LiquidAI/LFM3-Omni-2B running natively on Apple Silicon via MLX.
Starts a local FastAPI web server with a Browser Call Studio UI at http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import math
import os
import uuid
from typing import Any

from voice_agent import (
    EngineSettings,
    FlametreeSettings,
    PersonaConfig,
    ProductCardTier,
    ProductSummary,
    SUPPORTED_LANGUAGES,
    ToolSpec,
    VoiceAgentConfig,
)
from voice_agent.server import run_server

DEFAULT_FLAMETREE_AGENT_ID = "1717ee86-0823-4700-aa70-d2b2b1c165bb"
DEFAULT_FLAMETREE_TENANT_ID = "8470d927-5847-48fc-9a55-d72dd4117ac1"

# ---------------------------------------------------------------------------
# Karoo Crest Bank Product Catalog & Tariffs (ZAR)
# ---------------------------------------------------------------------------

TIER_TABLE: dict[str, dict[str, Any]] = {
    "classic": {
        "name": "Ubuntu Classic",
        "monthly_fee_zar": 55,
        "min_income_zar": 8000,
        "min_limit_zar": 5000,
        "max_limit_zar": 25000,
        "base_rate_pct": 19.5,
        "rate_range": "18.5% – 20.5% p.a.",
        "cashback": "2% cashback at Checkers, Pick n Pay & Engen",
    },
    "gold": {
        "name": "Ubuntu Gold",
        "monthly_fee_zar": 95,
        "min_income_zar": 20000,
        "min_limit_zar": 25000,
        "max_limit_zar": 65000,
        "base_rate_pct": 17.5,
        "rate_range": "16.5% – 18.5% p.a.",
        "cashback": "2.5% cashback at Woolworths, Checkers & Engen + 1 lounge visit/yr",
    },
    "protea_platinum": {
        "name": "Ubuntu Protea Platinum",
        "monthly_fee_zar": 145,
        "min_income_zar": 45000,
        "min_limit_zar": 65000,
        "max_limit_zar": 150000,
        "base_rate_pct": 15.5,
        "rate_range": "14.75% – 16.5% p.a.",
        "cashback": "3% cashback + 0% FX fees + 4 domestic SLOW/Bidvest lounge visits/yr",
    },
}


def _normalize_tier(raw_tier: str) -> str:
    cleaned = str(raw_tier or "gold").strip().lower().replace("-", "_").replace(" ", "_")
    if "plat" in cleaned or "protea" in cleaned:
        return "protea_platinum"
    if "classic" in cleaned:
        return "classic"
    return "gold"


# ---------------------------------------------------------------------------
# Live CRM & Banking Tool Implementations
# ---------------------------------------------------------------------------


def check_pre_qualification(
    monthly_income_zar: float,
    employment_type: str = "full_time",
    has_sa_id: bool = True,
) -> dict[str, Any]:
    """Check South African NCA affordability tier and indicative ZAR credit limit."""
    try:
        income = float(monthly_income_zar)
    except (TypeError, ValueError):
        return {"ok": False, "eligible": False, "error": "Invalid monthly income value."}

    emp = str(employment_type or "full_time").strip().lower().replace("-", "_")
    sa_id = bool(has_sa_id)

    if not sa_id:
        return {
            "ok": True,
            "eligible": False,
            "reason": "A valid South African ID book or Smart ID card is required for FICA verification.",
            "summary": "Not eligible without a valid South African ID (FICA requirement).",
        }

    if emp in ("unemployed", "none", "student"):
        return {
            "ok": True,
            "eligible": False,
            "reason": "Regular verifiable income is required under the National Credit Act.",
            "summary": "Not eligible without verifiable monthly income under the NCA.",
        }

    if income < 8000:
        return {
            "ok": True,
            "eligible": False,
            "reason": f"Gross monthly income of R{income:,.0f} is below the R8,000 Classic minimum.",
            "summary": f"Income R{income:,.0f} is below the R8,000 minimum threshold.",
        }

    if income >= 45000:
        tier_key = "protea_platinum"
    elif income >= 20000:
        tier_key = "gold"
    else:
        tier_key = "classic"

    tier = TIER_TABLE[tier_key]
    multiplier = 1.6 if "self" in emp or "contract" in emp else 2.1
    raw_limit = round((income * multiplier) / 1000.0) * 1000
    est_limit = int(
        max(tier["min_limit_zar"], min(tier["max_limit_zar"], raw_limit))
    )

    return {
        "ok": True,
        "eligible": True,
        "tier_key": tier_key,
        "recommended_tier": tier["name"],
        "estimated_limit_zar": est_limit,
        "monthly_fee_zar": tier["monthly_fee_zar"],
        "indicative_rate_pct": tier["base_rate_pct"],
        "cashback_summary": tier["cashback"],
        "interest_free_days": 55,
        "summary": (
            f"Pre-qualified for {tier['name']} with an estimated limit of "
            f"R{est_limit:,} at {tier['base_rate_pct']}% p.a. (R{tier['monthly_fee_zar']}/month)."
        ),
    }


def calculate_monthly_repayment(
    amount_zar: float,
    months: int = 12,
    tier: str = "gold",
) -> dict[str, Any]:
    """Calculate amortized monthly repayment in ZAR for a given balance and term."""
    try:
        principal = max(500.0, float(amount_zar))
        term_months = max(1, min(60, int(months)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "Invalid amount or term."}

    tier_key = _normalize_tier(tier)
    info = TIER_TABLE[tier_key]
    annual_rate = float(info["base_rate_pct"])
    monthly_rate = (annual_rate / 100.0) / 12.0

    if monthly_rate > 0:
        payment = (
            principal
            * (monthly_rate * math.pow(1.0 + monthly_rate, term_months))
            / (math.pow(1.0 + monthly_rate, term_months) - 1.0)
        )
    else:
        payment = principal / term_months

    total_payable = payment * term_months
    total_interest = total_payable - principal

    return {
        "ok": True,
        "tier_key": tier_key,
        "tier_name": info["name"],
        "principal_zar": round(principal, 2),
        "months": term_months,
        "annual_rate_pct": annual_rate,
        "monthly_repayment_zar": round(payment, 2),
        "total_interest_zar": round(total_interest, 2),
        "total_payable_zar": round(total_payable, 2),
        "summary": (
            f"R{principal:,.0f} over {term_months} months on {info['name']} "
            f"({annual_rate}% p.a.) is R{payment:,.2f} per month."
        ),
    }


def submit_card_application(
    full_name: str,
    phone_number: str,
    tier: str = "gold",
    monthly_income_zar: float = 20000,
    popia_consent: bool = False,
) -> dict[str, Any]:
    """Submit a provisional credit card application after explicit POPIA consent."""
    name = str(full_name or "").strip()
    phone = str(phone_number or "").strip()
    if not popia_consent:
        return {
            "ok": False,
            "error": "Explicit POPIA consent is required before submitting an application.",
            "summary": "POPIA consent was not granted yet.",
        }
    if not name or not phone:
        return {
            "ok": False,
            "error": "Applicant full name and South African phone number are required.",
            "summary": "Missing applicant name or phone number.",
        }

    tier_key = _normalize_tier(tier)
    info = TIER_TABLE[tier_key]
    ref = f"KCB-{uuid.uuid4().hex[:6].upper()}"
    return {
        "ok": True,
        "reference_number": ref,
        "applicant_name": name,
        "phone_number": phone,
        "card_tier": info["name"],
        "monthly_income_zar": float(monthly_income_zar),
        "popia_consent": True,
        "status": "Provisional Approval Pending FICA",
        "summary": (
            f"Provisional application {ref} submitted for {name} ({info['name']}). "
            "SMS link sent for FICA ID & payslip upload."
        ),
    }


def schedule_callback(
    phone_number: str,
    preferred_time: str,
    notes: str = "",
) -> dict[str, Any]:
    """Book a follow-up call with a Karoo Crest Bank credit specialist."""
    phone = str(phone_number or "").strip()
    when = str(preferred_time or "tomorrow morning").strip()
    if not phone:
        return {
            "ok": False,
            "error": "A phone number is required to schedule a callback.",
            "summary": "Phone number missing for callback.",
        }
    cb_id = f"CB-{uuid.uuid4().hex[:5].upper()}"
    return {
        "ok": True,
        "callback_id": cb_id,
        "phone_number": phone,
        "preferred_time": when,
        "notes": str(notes or "").strip(),
        "summary": f"Callback {cb_id} booked for {phone} at {when}.",
    }


def build_tools() -> list[ToolSpec]:
    """Return the CRM tool definitions for Karoo Crest Bank."""
    return [
        ToolSpec(
            name="check_pre_qualification",
            description=(
                "Check which Ubuntu Rewards Credit Card tier and indicative ZAR limit a "
                "South African customer qualifies for based on gross monthly income, employment type, and SA ID status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "monthly_income_zar": {
                        "type": "number",
                        "description": "Gross monthly income in South African Rand (ZAR).",
                    },
                    "employment_type": {
                        "type": "string",
                        "enum": ["full_time", "self_employed", "contract", "pensioner", "unemployed"],
                        "description": "Customer employment status.",
                    },
                    "has_sa_id": {
                        "type": "boolean",
                        "description": "True if the customer holds a valid South African ID.",
                    },
                },
                "required": ["monthly_income_zar"],
            },
            handler=check_pre_qualification,
        ),
        ToolSpec(
            name="calculate_monthly_repayment",
            description="Calculate the estimated monthly repayment in ZAR for a card balance and repayment term in months.",
            parameters={
                "type": "object",
                "properties": {
                    "amount_zar": {
                        "type": "number",
                        "description": "Balance amount in ZAR.",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Repayment period in months (1 to 60).",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["classic", "gold", "protea_platinum"],
                        "description": "Ubuntu Rewards Card tier.",
                    },
                },
                "required": ["amount_zar", "months"],
            },
            handler=calculate_monthly_repayment,
        ),
        ToolSpec(
            name="submit_card_application",
            description=(
                "Submit a provisional Ubuntu Rewards Credit Card application ONLY after the customer gives explicit POPIA consent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "description": "Applicant full legal name."},
                    "phone_number": {"type": "string", "description": "South African mobile number."},
                    "tier": {
                        "type": "string",
                        "enum": ["classic", "gold", "protea_platinum"],
                        "description": "Chosen Ubuntu Rewards Card tier.",
                    },
                    "monthly_income_zar": {"type": "number", "description": "Gross monthly income in ZAR."},
                    "popia_consent": {
                        "type": "boolean",
                        "description": "Must be true only after customer explicitly agrees to POPIA credit check.",
                    },
                },
                "required": ["full_name", "phone_number", "tier", "popia_consent"],
            },
            handler=submit_card_application,
        ),
        ToolSpec(
            name="schedule_callback",
            description="Schedule a follow-up phone call if the customer asks to think about it or call later.",
            parameters={
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Customer phone number."},
                    "preferred_time": {"type": "string", "description": "Preferred date and time window."},
                    "notes": {"type": "string", "description": "Optional context notes for the specialist."},
                },
                "required": ["phone_number", "preferred_time"],
            },
            handler=schedule_callback,
        ),
    ]


# ---------------------------------------------------------------------------
# Sales Personas & Voice-First System Prompts
# ---------------------------------------------------------------------------


def _build_system_prompt(agent_name: str, gender: str) -> str:
    return (
        f"You are {agent_name}, a warm, natural South African credit card telesales specialist "
        f"calling from Karoo Crest Bank in Stellenbosch and Johannesburg (FSP 51842, NCRCP 14892). "
        f"Speak using a {gender} voice.\n\n"
        "PRODUCT — Karoo Crest Ubuntu Rewards Credit Card (South African Rand / ZAR):\n"
        "- Ubuntu Classic: R55/month fee, credit limit R5,000 to R25,000, minimum R8,000 gross monthly income, "
        "2% cashback at Checkers, Pick n Pay, and Engen, interest 18.5% to 20.5% p.a.\n"
        "- Ubuntu Gold: R95/month fee, credit limit R25,000 to R65,000, minimum R20,000 gross monthly income, "
        "2.5% cashback at Woolworths, Checkers, and Engen plus 1 airport lounge visit per year, interest 16.5% to 18.5% p.a.\n"
        "- Ubuntu Protea Platinum: R145/month fee, credit limit R65,000 to R150,000, minimum R45,000 gross monthly income, "
        "3% cashback, zero foreign exchange fees, 4 domestic airport lounge visits per year, interest 14.75% to 16.5% p.a.\n"
        "- Every card includes up to 55 days interest-free on purchases, instant virtual cards, and 24/7 Cape Town support.\n\n"
        "STRICT CONVERSATION RULES:\n"
        f"1. Always stay in character as {agent_name} (the bank advisor); address the caller politely and wait for them to state their own income, spend, name, or phone number.\n"
        "2. Advance the conversation forward every turn; never repeat earlier introductions or availability checks.\n"
        "3. When the caller says yes, please continue, or go ahead, briefly highlight one or two benefits (such as up to 55 days interest-free and up to 3 percent cashback on groceries and fuel) and ask one discovery question.\n"
        "4. If the caller has not yet answered with a number or detail, never assume one on their behalf; share another perk and ask gently.\n"
        "5. Keep every spoken reply concise (2 short sentences max), zero markdown, zero bullet lists, zero emojis.\n"
        "6. Call check_pre_qualification only after the caller states their income; call submit_card_application only after explicit POPIA consent; offer schedule_callback if busy.\n\n"
        "Respond with function calls, and interleaved text and audio."
    )


def build_personas() -> list[PersonaConfig]:
    return [
        PersonaConfig(
            id="naledi",
            name="Naledi Mokoena",
            role="Senior Card Specialist · Johannesburg",
            voice_gender="female",
            voice_accent="British English",
            description="Warm, energetic South African telesales advisor (female voice).",
            greeting_trigger=(
                "The customer just answered your outbound call. Deliver a warm, brief "
                "15-second opening greeting introducing yourself as Naledi from Karoo Crest Bank, "
                "mentioning the Ubuntu Rewards Credit Card, and politely asking if they have a quick minute."
            ),
            system_prompt=_build_system_prompt("Naledi Mokoena", "female"),
        ),
        PersonaConfig(
            id="thabo",
            name="Thabo Dlamini",
            role="Credit Card Advisor · Cape Town",
            voice_gender="male",
            voice_accent="British English",
            description="Calm, consultative South African telesales advisor (male voice).",
            greeting_trigger=(
                "The customer just answered your outbound call. Deliver a warm, brief "
                "15-second opening greeting introducing yourself as Thabo from Karoo Crest Bank, "
                "mentioning the Ubuntu Rewards Credit Card, and politely asking if they have a quick minute."
            ),
            system_prompt=_build_system_prompt("Thabo Dlamini", "male"),
        ),
    ]


def build_product_summary() -> ProductSummary:
    return ProductSummary(
        product_name="Ubuntu Rewards Credit Card",
        tagline="Boutique South African rewards credit card with up to 55 days interest-free",
        currency="ZAR",
        interest_free_days=55,
        compliance_note=(
            "Karoo Crest Bank (Pty) Ltd is a fictional South African boutique bank demo. "
            "Regulated demo posture: FSP #51842 · NCRCP14892 · POPIA & FICA verification required."
        ),
        tiers=[
            ProductCardTier(
                id="classic",
                name="Ubuntu Classic",
                monthly_fee="R55 / mo",
                limit_range="R5,000 – R25,000",
                min_income="R8,000 / mo",
                interest_rate="18.5% – 20.5% p.a.",
                cashback="2% Checkers, Pick n Pay & Engen",
                highlights=["55 days interest-free", "Free virtual cards", "24/7 Cape Town support"],
            ),
            ProductCardTier(
                id="gold",
                name="Ubuntu Gold",
                monthly_fee="R95 / mo",
                limit_range="R25,000 – R65,000",
                min_income="R20,000 / mo",
                interest_rate="16.5% – 18.5% p.a.",
                cashback="2.5% Woolworths, Checkers & Engen",
                highlights=["1 domestic lounge visit/yr", "Purchase protection", "Tap-to-pay & Apple Pay"],
            ),
            ProductCardTier(
                id="protea_platinum",
                name="Protea Platinum",
                monthly_fee="R145 / mo",
                limit_range="R65,000 – R150,000",
                min_income="R45,000 / mo",
                interest_rate="14.75% – 16.5% p.a.",
                cashback="3% partner cashback",
                highlights=["0% FX conversion fees", "4 domestic lounge visits/yr", "Priority concierge"],
            ),
        ],
    )


def build_config(
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: str = "mlx",
    model_id: str | None = None,
    model_revision: str | None = None,
    vllm_deploy_config: str | None = None,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    default_voice: str = "naledi",
    max_slots: int = 900,
    compaction_tokens: int = 4096,
    patience: str = "balanced",
    language: str = "en",
    delegation_mode: str = "selective_tool",
    dynamic_filler: bool = True,
    flametree_agent_id: str | None = None,
    flametree_base_url: str | None = None,
    calls_dir: str | None = None,
    record_call_audio: bool = True,
) -> VoiceAgentConfig:
    resolved_backend = (backend or os.environ.get("VOICE_BACKEND", "mlx")).strip().lower()
    if resolved_backend not in ("mlx", "vllm"):
        raise ValueError(f"Unsupported backend {resolved_backend!r}; expected 'mlx' or 'vllm'.")
    resolved_model_id = (
        model_id
        or os.environ.get("MODEL_ID", "").strip()
        or (
            "LiquidAI/LFM3-Omni-2B"
            if resolved_backend == "vllm"
            else "LiquidAI/LFM3-Omni-2B-MLX-8bit"
        )
    )
    resolved_model_revision = (
        model_revision
        or os.environ.get("MODEL_REVISION", "").strip()
        or "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"
    )
    resolved_vllm_deploy_config = (
        vllm_deploy_config
        or os.environ.get("DEPLOY_CONFIG", "").strip()
        or "deploy/lfm3_omni_runpod.yaml"
    )
    resolved_vllm_url = (
        vllm_base_url
        or os.environ.get("VLLM_BASE_URL", "").strip()
        or ""
    )
    resolved_vllm_key = vllm_api_key or os.environ.get("VLLM_API_KEY", "").strip() or None

    voice_map = {"female": "naledi", "male": "thabo"}
    persona_id = voice_map.get(default_voice.lower(), default_voice.lower())
    agent_id = (
        flametree_agent_id
        or os.environ.get("FLAMETREE_AGENT_ID", "").strip()
        or DEFAULT_FLAMETREE_AGENT_ID
    )
    base_url = (
        flametree_base_url
        or os.environ.get("FLAMETREE_BASE_URL", "").strip()
        or "https://portal.flametree.ai"
    )
    if calls_dir is None:
        raw_calls_dir = os.environ.get("CALLS_DIR", "calls").strip()
        resolved_calls_dir: str | None = raw_calls_dir or None
    else:
        resolved_calls_dir = calls_dir.strip() or None

    return VoiceAgentConfig(
        app_title="Karoo Crest Bank — Pure-S2S Voice Call Studio",
        company_name="Karoo Crest Bank",
        subtitle="Ubuntu Rewards Credit Card · Multilingual Pure-S2S + Flametree Agent Delegation",
        badges=[
            "Stellenbosch · JHB",
            "FSP #51842",
            "NCRCP 14892",
            "es-MX · es · en",
            "Flametree Agent",
        ],
        personas=build_personas(),
        default_persona_id=persona_id,
        tools=build_tools(),
        product_summary=build_product_summary(),
        engine=EngineSettings(
            backend=resolved_backend,
            model_id=resolved_model_id,
            model_revision=resolved_model_revision,
            vllm_deploy_config=resolved_vllm_deploy_config,
            vllm_base_url=resolved_vllm_url,
            vllm_api_key=resolved_vllm_key,
            max_slots=max_slots,
            compaction_trigger_tokens=compaction_tokens,
        ),
        host=host,
        port=port,
        default_turn_patience=patience,
        target_call_duration_sec=300,
        default_language=language,
        supported_languages=list(SUPPORTED_LANGUAGES),
        default_delegation_mode=delegation_mode,
        default_dynamic_filler=dynamic_filler,
        flametree=FlametreeSettings(
            enabled=True,
            base_url=base_url,
            environment="prod",
            tenant_id=DEFAULT_FLAMETREE_TENANT_ID,
            tenant_name="Denis Demo",
            agent_id=agent_id,
        ),
        calls_dir=resolved_calls_dir,
        record_call_audio=record_call_audio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Karoo Crest Bank South African Credit Card Telesales Voice Agent."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/WS port (default: 8000)")
    parser.add_argument(
        "--backend",
        default=os.environ.get("VOICE_BACKEND", "mlx"),
        choices=["mlx", "vllm"],
        help="Inference backend: 'mlx' (macOS Apple Silicon) or 'vllm' (native vLLM-Omni AsyncOmni)",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="HuggingFace model revision for --backend vllm",
    )
    parser.add_argument(
        "--vllm-deploy-config",
        default=None,
        help="Path to vLLM-Omni deploy YAML when --backend vllm (default: deploy/lfm3_omni_runpod.yaml)",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_BASE_URL"),
        help="Deprecated compatibility flag (vllm backend now runs native AsyncOmni in-process)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (defaults to LFM3-Omni-2B-MLX-8bit on mlx, LFM3-Omni-2B on vllm)",
    )
    parser.add_argument(
        "--voice",
        default="naledi",
        choices=["naledi", "thabo", "female", "male"],
        help="Default telesales persona (default: naledi)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=list(SUPPORTED_LANGUAGES),
        help="Default call language: es-MX, es, or en (default: en)",
    )
    parser.add_argument(
        "--delegation-mode",
        default="selective_tool",
        choices=["selective_tool", "always_delegate"],
        help="Knowledge delegation strategy (default: selective_tool)",
    )
    parser.add_argument(
        "--no-filler",
        action="store_true",
        help="Disable dynamic MLX B=2 parallel voice fillers during delegation",
    )
    parser.add_argument(
        "--flametree-agent-id",
        default=None,
        help="Override Flametree agent UUID (default: Denis Demo Karoo Crest Knowledge Expert)",
    )
    parser.add_argument(
        "--flametree-base-url",
        default=None,
        help="Override Flametree portal URL (default: https://portal.flametree.ai)",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=900,
        help="Maximum generated slots per turn (default: 900)",
    )
    parser.add_argument(
        "--compaction-tokens",
        type=int,
        default=4096,
        help="Context token threshold before rolling conversation compaction (default: 4096)",
    )
    parser.add_argument(
        "--patience",
        default="balanced",
        choices=["snappy", "balanced", "patient"],
        help="Model-driven turn-end patience preset (default: balanced)",
    )
    parser.add_argument(
        "--calls-dir",
        default=None,
        help="Directory for per-call artifacts (default: CALLS_DIR env var or ./calls; '' disables)",
    )
    parser.add_argument(
        "--no-record-audio",
        action="store_true",
        help="Disable saving caller_16k.wav and agent_24k.wav inside each call folder",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the Browser Call Studio automatically after startup",
    )
    args = parser.parse_args()

    config = build_config(
        host=args.host,
        port=args.port,
        backend=args.backend,
        model_id=args.model,
        model_revision=args.model_revision,
        vllm_deploy_config=args.vllm_deploy_config,
        vllm_base_url=args.vllm_url,
        default_voice=args.voice,
        max_slots=args.max_slots,
        compaction_tokens=args.compaction_tokens,
        patience=args.patience,
        language=args.lang,
        delegation_mode=args.delegation_mode,
        dynamic_filler=not args.no_filler,
        flametree_agent_id=args.flametree_agent_id,
        flametree_base_url=args.flametree_base_url,
        calls_dir=args.calls_dir,
        record_call_audio=not args.no_record_audio,
    )
    run_server(config, open_browser=args.open)


if __name__ == "__main__":
    main()
