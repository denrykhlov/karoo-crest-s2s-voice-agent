"""Unit tests for LFM3 XML tool parsing, barge-in truncation, and Karoo Crest Bank CRM tools."""

from __future__ import annotations

import numpy as np

from telesales_agent import (
    calculate_monthly_repayment,
    check_pre_qualification,
    schedule_callback,
    submit_card_application,
)
from voice_agent.engine import (
    TurnRecord,
    parse_tool_calls,
    render_tool_calls,
    truncate_assistant_turn_for_barge_in,
)


def test_parse_and_render_tool_calls() -> None:
    raw = """<tool_call>
<function=check_pre_qualification>
<parameter=monthly_income_zar>
28000
</parameter>
<parameter=employment_type>
full_time
</parameter>
<parameter=has_sa_id>
true
</parameter>
</function>
</tool_call>"""

    calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "check_pre_qualification"
    args = calls[0]["function"]["arguments"]
    assert args["monthly_income_zar"] == 28000
    assert args["employment_type"] == "full_time"
    assert args["has_sa_id"] is True

    rerendered = render_tool_calls(calls)
    assert parse_tool_calls(rerendered) == calls


def test_parse_multiple_tool_calls_and_malformed() -> None:
    raw = (
        "ignore <tool_call><function=broken></tool_call>\n"
        "<tool_call>\n<function=calculate_monthly_repayment>\n"
        "<parameter=amount_zar>\n15000\n</parameter>\n"
        "<parameter=months>\n12\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=schedule_callback>\n"
        "<parameter=phone_number>\n+27825551234\n</parameter>\n"
        "<parameter=preferred_time>\nTomorrow at 10am\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(raw)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "calculate_monthly_repayment"
    assert calls[1]["function"]["arguments"]["phone_number"] == "+27825551234"


def test_karoo_crest_pre_qualification_tiers() -> None:
    no_id = check_pre_qualification(35000, "full_time", has_sa_id=False)
    assert no_id["ok"] and not no_id["eligible"]

    low = check_pre_qualification(5500, "full_time", has_sa_id=True)
    assert low["ok"] and not low["eligible"]

    classic = check_pre_qualification(12000, "full_time", has_sa_id=True)
    assert classic["eligible"] and classic["tier_key"] == "classic"
    assert 5000 <= classic["estimated_limit_zar"] <= 25000

    gold = check_pre_qualification(28000, "full_time", has_sa_id=True)
    assert gold["eligible"] and gold["tier_key"] == "gold"
    assert 25000 <= gold["estimated_limit_zar"] <= 65000

    plat = check_pre_qualification(60000, "full_time", has_sa_id=True)
    assert plat["eligible"] and plat["tier_key"] == "protea_platinum"
    assert plat["estimated_limit_zar"] == 126000


def test_karoo_crest_repayment_application_and_callback() -> None:
    calc = calculate_monthly_repayment(amount_zar=20000, months=12, tier="gold")
    assert calc["ok"]
    assert calc["annual_rate_pct"] == 17.5
    # Exact standard annuity payment for R20,000 at 17.5%/12 for 12m is ~R1,829.26
    assert 1820.0 < calc["monthly_repayment_zar"] < 1840.0
    assert calc["total_interest_zar"] > 0

    no_popia = submit_card_application(
        full_name="Sipho Ndlovu",
        phone_number="0825550199",
        tier="gold",
        monthly_income_zar=28000,
        popia_consent=False,
    )
    assert not no_popia["ok"]

    app_ok = submit_card_application(
        full_name="Sipho Ndlovu",
        phone_number="0825550199",
        tier="gold",
        monthly_income_zar=28000,
        popia_consent=True,
    )
    assert app_ok["ok"]
    assert app_ok["reference_number"].startswith("KCB-")
    assert app_ok["popia_consent"] is True

    cb = schedule_callback(phone_number="0825550199", preferred_time="Friday 14:00")
    assert cb["ok"] and cb["callback_id"].startswith("CB-")


class _DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(x) for x in ids if x < 1000)


def test_barge_in_truncation_keeps_exact_placeholder_alignment() -> None:
    audio_out_id = 125032
    eos_code = 2016
    im_end_id = 125001

    # 3 text tokens, 10 speech frames + 1 eos frame + im_end
    raw_ids = [10, 11, 12] + [audio_out_id] * 10 + [13, 14] + [audio_out_id] + [im_end_id]
    codes = np.zeros((11, 12), dtype=np.int32)
    codes[:10, :] = np.arange(10)[:, None] + 1
    codes[10, :] = eos_code

    turn = TurnRecord(
        role="assistant",
        text="Full message",
        raw_token_ids=raw_ids,
        raw_audio_codes=codes,
    )
    truncated = truncate_assistant_turn_for_barge_in(
        turn,
        played_frames=4,
        audio_out_token_id=audio_out_id,
        audio_eos_code=eos_code,
        im_end_id=im_end_id,
        tokenizer=_DummyTokenizer(),
    )

    assert truncated.raw_token_ids is not None
    assert truncated.raw_audio_codes is not None
    placeholder_count = sum(1 for t in truncated.raw_token_ids if t == audio_out_id)
    assert placeholder_count == truncated.raw_audio_codes.shape[0] == 5  # 4 kept + 1 EOS
    assert np.all(truncated.raw_audio_codes[-1] == eos_code)
    assert truncated.raw_token_ids[-1] == im_end_id
    assert truncated.text.endswith("...")
