"""Flametree Platform delegation client and offline banking KB fallback."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .config import FlametreeSettings
from .languages import normalize_language_code

_MARKDOWN_STRIP_RE = re.compile(r"[*_`#>]+")


def _clean_for_voice(text: str) -> str:
    """Normalize remote agent text so local S2S speaks naturally without markdown artifacts."""
    cleaned = _MARKDOWN_STRIP_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_session_id(data: Any) -> Optional[str]:
    """Extract runtime session UUID from Flametree `POST /session` responses."""
    if isinstance(data, str):
        val = data.strip()
        return val or None
    if not isinstance(data, dict):
        return None
    for key in ("session_id", "id"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    for container_key in ("session", "data", "result"):
        nested = data.get(container_key)
        if isinstance(nested, dict):
            for key in ("session_id", "id"):
                raw = nested.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
    return None


def _extract_reply_text(data: Any) -> str:
    """Extract assistant response text from Flametree `POST .../run_v2` JSON."""
    if not isinstance(data, dict):
        return str(data or "").strip()
    for key in ("response", "message", "text"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("response", "message", "text"):
            raw = nested.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return ""


@dataclass(slots=True)
class DelegationResult:
    """Structured result returned by a 3rd-party knowledge delegate."""

    answer: str
    source: str  # "flametree_live" | "local_kb_fallback" | "error_fallback"
    latency_ms: float
    query: str = ""
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None


class KnowledgeDelegate:
    """Async interface for delegating complex banking knowledge queries."""

    @property
    def is_live(self) -> bool:
        return False

    @property
    def current_session_id(self) -> Optional[str]:
        return None

    @property
    def last_error(self) -> Optional[str]:
        return None

    async def start_session(self, call_id: str, language: str) -> Optional[str]:
        return None

    async def query(
        self,
        question: str,
        language: str,
        context_hint: str = "",
    ) -> DelegationResult:
        raise NotImplementedError

    async def close_session(self) -> None:
        return None


class LocalKnowledgeFallbackDelegate(KnowledgeDelegate):
    """Offline structured banking knowledge fallback when live Flametree credentials are unset."""

    def __init__(self, reason: str = "no_flametree_token") -> None:
        self.reason = reason
        self._last_error: Optional[str] = (
            None if reason == "no_flametree_token" else reason
        )

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def query(
        self,
        question: str,
        language: str,
        context_hint: str = "",
    ) -> DelegationResult:
        t0 = time.perf_counter()
        lang = normalize_language_code(language)
        q = (question or "").lower()
        answer = self._lookup_answer(q, lang)
        dt = (time.perf_counter() - t0) * 1000.0
        return DelegationResult(
            answer=answer,
            source="local_kb_fallback",
            latency_ms=dt,
            query=question,
            error=self._last_error,
        )

    @staticmethod
    def _lookup_answer(q: str, lang: str) -> str:
        # FX / foreign / divisas / extranjero
        if any(k in q for k in ("fx", "foreign", "divisa", "extranjero", "tipo de cambio", "internacional", "currency")):
            if lang == "es-MX":
                return (
                    "La tarjeta Ubuntu Protea Platinum tiene cero por ciento de comisión por cambio de divisa en el extranjero. "
                    "Las tarjetas Ubuntu Classic y Ubuntu Gold aplican un cargo de dos punto setenta y cinco por ciento por conversión de divisas."
                )
            if lang == "es":
                return (
                    "La tarjeta Ubuntu Protea Platinum ofrece cero por ciento de comisión por cambio de divisa. "
                    "En Ubuntu Classic y Ubuntu Gold el margen por operaciones en moneda extranjera es del dos coma setenta y cinco por ciento."
                )
            return (
                "Ubuntu Protea Platinum has zero percent foreign exchange fees worldwide, while Ubuntu Classic and "
                "Ubuntu Gold carry a 2.75 percent currency conversion markup on international purchases."
            )

        # Lounge / VIP / viaje / seguro / airport
        if any(k in q for k in ("lounge", "sala", "aeropuerto", "vip", "viaje", "travel", "insurance", "seguro")):
            if lang == "es-MX":
                return (
                    "Ubuntu Protea Platinum incluye cuatro visitas al año a salas VIP nacionales SLOW o Bidvest y seguro de viaje hasta por tres millones de rands al pagar los boletos con la tarjeta. "
                    "Ubuntu Gold incluye una visita nacional al año, y Ubuntu Classic no incluye acceso a salas."
                )
            if lang == "es":
                return (
                    "Ubuntu Protea Platinum incluye cuatro accesos anuales a salas nacionales SLOW o Bidvest y seguro de viaje de hasta tres millones de rands al comprar los billetes con la tarjeta. "
                    "Ubuntu Gold incluye una visita anual en Sudáfrica."
                )
            return (
                "Ubuntu Protea Platinum includes four complimentary domestic SLOW or Bidvest lounge visits per year plus travel insurance up to three million Rand when flights are bought on the card. "
                "Ubuntu Gold includes one domestic lounge visit per year."
            )

        # FICA / POPIA / NCA / requisitos / documents / ID / eligibility
        if any(k in q for k in ("fica", "popia", "nca", "requisit", "document", "eligib", "qualif", "identif", "nómina", "nomina", "payslip", "comprobante")):
            if lang == "es-MX":
                return (
                    "Para aprobación bajo la Ley Nacional de Crédito y FICA necesitas identificación sudafricana vigente, tus últimos tres recibos de nómina o seis meses de estados de cuenta si trabajas por tu cuenta, y comprobante de domicilio menor a tres meses. "
                    "Además requerimos tu consentimiento explícito POPIA antes de realizar cualquier consulta suave de crédito."
                )
            if lang == "es":
                return (
                    "Bajo la normativa NCA y FICA necesitamos un documento de identidad sudafricano válido, las tres últimas nóminas o seis meses de extractos bancarios para autónomos, y justificante de domicilio reciente de menos de tres meses. "
                    "Siempre solicitamos tu consentimiento expreso POPIA antes de cualquier consulta crediticia."
                )
            return (
                "Under NCA affordability and FICA rules, you need a valid South African ID, your latest three months of payslips or six months of bank statements if self-employed, and proof of residence under three months old. "
                "Explicit POPIA consent is always required before any soft credit check."
            )

        # 55 days grace period / interest-free / adelanto / efectivo / mora / iniciación
        if any(k in q for k in ("55", "grace", "sin intereses", "interest-free", "cash advance", "efectivo", "adelanto", "late", "mora", "initiation", "apertura")):
            if lang == "es-MX":
                return (
                    "Todas las tarjetas ofrecen hasta cincuenta y cinco días sin intereses en compras en comercios o en línea pagando el saldo total antes de la fecha límite, sin aplicar a disposiciones de efectivo ni transferencias. "
                    "La comisión por retiro o adelanto de efectivo es de dos punto cinco por ciento con mínimo de cuarenta y cinco rands, y la comisión de apertura de ciento sesenta y cinco rands está exenta en esta promoción."
                )
            if lang == "es":
                return (
                    "Dispones de hasta cincuenta y cinco días sin intereses en compras si liquidas el saldo total del extracto a tiempo, quedando excluidos los adelantos en efectivo y transferencias. "
                    "La comisión por disposición de efectivo es del dos coma cinco por ciento con mínimo de cuarenta y cinco rands."
                )
            return (
                "All Ubuntu cards include up to 55 days interest-free on retail and online purchases when you pay the full statement balance by the due date, excluding cash advances and EFTs. "
                "Cash advances carry a 2.5 percent fee with a 45 Rand minimum, and the 165 Rand initiation fee is waived on our promo."
            )

        # Platinum specifically
        if any(k in q for k in ("plat", "protea", "145")):
            if lang == "es-MX":
                return (
                    "Ubuntu Protea Platinum tiene una cuota mensual de ciento cuarenta y cinco rands, límite de sesenta y cinco mil a ciento cincuenta mil rands para ingresos desde cuarenta y cinco mil rands al mes, tasa de catorce punto setenta y cinco a dieciséis punto cinco por ciento anual, tres por ciento de cashback y cero por ciento de comisión en el extranjero."
                )
            if lang == "es":
                return (
                    "Ubuntu Protea Platinum cuesta ciento cuarenta y cinco rands al mes, ofrece límite entre sesenta y cinco mil y ciento cincuenta mil rands desde cuarenta y cinco mil rands de ingreso mensual, tasa del catorce coma setenta y cinco al dieciséis coma cinco por ciento anual, tres por ciento de reembolso y cero comisión de cambio de divisa."
                )
            return (
                "Ubuntu Protea Platinum costs 145 Rand per month for clients earning 45,000 Rand or more, with a limit up to 150,000 Rand, 14.75 to 16.5 percent annual interest, 3 percent partner cashback, zero FX fees, and four domestic lounge visits per year."
            )

        # Gold specifically
        if any(k in q for k in ("gold", "oro", "95")):
            if lang == "es-MX":
                return (
                    "Ubuntu Gold tiene cuota mensual de noventa y cinco rands, límite de veinticinco mil a sesenta y cinco mil rands para ingresos desde veinte mil rands mensuales, tasa de dieciséis punto cinco a dieciocho punto cinco por ciento anual, dos punto cinco por ciento de cashback en Woolworths, Checkers y Engen, más una visita nacional a sala VIP al año."
                )
            if lang == "es":
                return (
                    "Ubuntu Gold tiene una cuota de noventa y cinco rands al mes, límite de veinticinco mil a sesenta y cinco mil rands desde veinte mil rands de ingresos, tasa del dieciséis coma cinco al dieciocho coma cinco por ciento anual y dos coma cinco por ciento de reembolso más un acceso nacional a sala VIP."
                )
            return (
                "Ubuntu Gold has a 95 Rand monthly fee for incomes from 20,000 Rand, limits from 25,000 to 65,000 Rand, 16.5 to 18.5 percent interest, 2.5 percent cashback at Woolworths, Checkers, and Engen, plus one domestic lounge visit per year."
            )

        # Classic / general comparison
        if lang == "es-MX":
            return (
                "En Karoo Crest Bank ofrecemos Ubuntu Classic por cincuenta y cinco rands al mes desde ocho mil rands de ingreso, Ubuntu Gold por noventa y cinco rands desde veinte mil rands, y Protea Platinum por ciento cuarenta y cinco rands desde cuarenta y cinco mil rands. "
                "Todas incluyen hasta cincuenta y cinco días sin intereses y cashback de dos a tres por ciento."
            )
        if lang == "es":
            return (
                "Karoo Crest Bank ofrece Ubuntu Classic por cincuenta y cinco rands al mes desde ocho mil rands de ingresos, Ubuntu Gold por noventa y cinco rands desde veinte mil rands y Protea Platinum por ciento cuarenta y cinco rands desde cuarenta y cinco mil rands, todas con hasta cincuenta y cinco días sin intereses y reembolso de hasta el tres por ciento."
            )
        return (
            "Karoo Crest Bank offers Ubuntu Classic at 55 Rand per month from 8,000 Rand income, Ubuntu Gold at 95 Rand from 20,000 Rand income, and Protea Platinum at 145 Rand from 45,000 Rand income. "
            "All three include up to 55 days interest-free on purchases, instant virtual cards, and 2 to 3 percent cashback."
        )


class FlametreeHttpDelegate(KnowledgeDelegate):
    """Live HTTP client integrating with the Flametree Chat API per phone call."""

    def __init__(
        self,
        settings: FlametreeSettings,
        api_token: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        raw_token = (
            api_token
            if api_token is not None
            else os.environ.get(settings.api_token_env, "")
        )
        cleaned_token = str(raw_token or "").strip()
        if cleaned_token.lower().startswith("bearer "):
            cleaned_token = cleaned_token[7:].strip()
        self._api_token = cleaned_token
        self._agent_route = (
            f"{settings.base_url.rstrip('/')}/chatbot/{settings.agent_id}"
        )
        headers: dict[str, str] = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        self._client = httpx.AsyncClient(
            timeout=settings.timeout_sec,
            headers=headers,
            transport=transport,
        )
        self._fallback = LocalKnowledgeFallbackDelegate(reason="flametree_http_fallback")
        self.session_id: Optional[str] = None
        self._last_error: Optional[str] = None
        self._session_lock = asyncio.Lock()

    @property
    def is_live(self) -> bool:
        return bool(self.settings.enabled and self.settings.agent_id and self._api_token)

    @property
    def current_session_id(self) -> Optional[str]:
        return self.session_id

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def start_session(self, call_id: str, language: str) -> Optional[str]:
        """Open one remote Flametree runtime session for this voice call."""
        if not self.is_live:
            return None
        lang = normalize_language_code(language)
        async with self._session_lock:
            if self.session_id:
                return self.session_id
            payload = {
                "user_id": call_id,
                "name": f"Voice Caller ({call_id[-8:]})",
                "type": "web",
                "args": [],
                "kwargs": {
                    "language": lang,
                    "channel": "lfm3-s2s-voice",
                },
            }
            try:
                resp = await self._client.post(
                    f"{self._agent_route}/session",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                sid = _extract_session_id(data)
                if sid:
                    self.session_id = sid
                    self._last_error = None
                    print(
                        f"[Flametree] Opened session {sid} for {call_id} (lang={lang})",
                        flush=True,
                    )
                    return sid
                keys_desc = (
                    list(data.keys())
                    if isinstance(data, dict)
                    else type(data).__name__
                )
                self.session_id = None
                self._last_error = (
                    f"Missing session_id in POST /session response (keys={keys_desc})"
                )
                print(f"[Flametree] {self._last_error}", flush=True)
                return None
            except Exception as exc:
                self.session_id = None
                self._last_error = f"POST /session failed: {exc}"
                print(f"[Flametree] {self._last_error}", flush=True)
                return None

    async def query(
        self,
        question: str,
        language: str,
        context_hint: str = "",
    ) -> DelegationResult:
        """Send one knowledge turn via `POST {agentRoute}/session/{sessionId}/run_v2`."""
        lang = normalize_language_code(language)
        if not self.is_live:
            return await self._fallback.query(question, lang, context_hint=context_hint)

        t0 = time.perf_counter()
        if not self.session_id:
            await self.start_session(f"voice-{uuid.uuid4().hex[:8]}", lang)
        if not self.session_id:
            res = await self._fallback.query(question, lang, context_hint=context_hint)
            res.error = self._last_error or "Failed to initialize remote Flametree session"
            res.latency_ms = (time.perf_counter() - t0) * 1000.0
            return res

        prefixed = f"[Language: {lang}] {question.strip()}"
        try:
            # Official Flametree Chat API requires multipart/form-data on run_v2.
            resp = await self._client.post(
                f"{self._agent_route}/session/{self.session_id}/run_v2",
                files={"message": (None, prefixed)},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            raw_reply = _extract_reply_text(data)
            cleaned = _clean_for_voice(raw_reply)
            dt = (time.perf_counter() - t0) * 1000.0
            if not cleaned:
                self._last_error = "Empty response from Flametree"
                fb = await self._fallback.query(question, lang, context_hint=context_hint)
                fb.error = self._last_error
                fb.latency_ms = dt
                return fb
            self._last_error = None
            print(
                f"[Flametree] run_v2 ok session={self.session_id} latency={dt:.0f}ms",
                flush=True,
            )
            return DelegationResult(
                answer=cleaned,
                source="flametree_live",
                latency_ms=dt,
                query=question,
                session_id=self.session_id,
                agent_id=self.settings.agent_id,
            )
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            self._last_error = f"run_v2 failed: {exc}"
            print(f"[Flametree] {self._last_error}", flush=True)
            fb = await self._fallback.query(question, lang, context_hint=context_hint)
            fb.error = str(exc)
            fb.latency_ms = dt
            return fb

    async def close_session(self) -> None:
        """Close the remote Flametree session cleanly when the phone call ends."""
        try:
            sid = self.session_id
            if sid and self._api_token:
                try:
                    await self._client.delete(
                        f"{self._agent_route}/session/{sid}",
                        params={"status": "completed"},
                    )
                    print(f"[Flametree] Closed session {sid}", flush=True)
                except Exception as exc:
                    print(f"[Flametree] close_session warning: {exc}", flush=True)
                finally:
                    self.session_id = None
        finally:
            await self._client.aclose()


def build_delegate(settings: FlametreeSettings) -> KnowledgeDelegate:
    """Create the appropriate knowledge delegate from runtime config/environment."""
    if settings.enabled and settings.agent_id and settings.has_token():
        return FlametreeHttpDelegate(settings)
    return LocalKnowledgeFallbackDelegate(reason="no_flametree_token")
