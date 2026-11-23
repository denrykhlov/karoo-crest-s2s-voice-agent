(() => {
  "use strict";

  const state = {
    config: null,
    ws: null,
    callActive: false,
    callState: "idle",
    muted: false,
    inputMode: "live", // "live" | "ptt"
    acceptingAudio: false,
    audioCtx: null,
    micStream: null,
    captureNode: null,
    playbackNode: null,
    ringInterval: null,
    timerInterval: null,
    callStartTs: 0,
    targetSec: 300,
    micRms: 0,
    agentRms: 0,
    turns: new Map(),
    pendingMetrics: new Map(),
    transcriptLog: [],
    delegationLog: [],
    latestDeliveryStats: null,
    crmToolCount: 0,
    callId: null,
    callArtifactsDir: "",
  };

  const els = {
    companyName: document.getElementById("company-name"),
    subtitle: document.getElementById("app-subtitle"),
    badgeList: document.getElementById("badge-list"),
    flametreePill: document.getElementById("flametree-pill"),
    pipelinePill: document.getElementById("pipeline-pill"),
    modelPill: document.getElementById("model-pill"),
    connPill: document.getElementById("conn-pill"),
    personaName: document.getElementById("active-persona-name"),
    personaRole: document.getElementById("active-persona-role"),
    callTimer: document.getElementById("call-timer"),
    callIdReadout: document.getElementById("call-id-readout"),
    callStateLabel: document.getElementById("call-state-label"),
    s2sIndicator: document.getElementById("s2s-indicator"),
    teleTtfaPerceived: document.getElementById("tele-ttfa-perceived"),
    teleTtfa: document.getElementById("tele-ttfa"),
    teleBargeTtfa: document.getElementById("tele-barge-ttfa"),
    teleSpeed: document.getElementById("tele-speed"),
    ctxReadout: document.getElementById("ctx-readout"),
    ctxBarFill: document.getElementById("ctx-bar-fill"),
    listenerReadout: document.getElementById("listener-readout"),
    btnCall: document.getElementById("btn-call"),
    btnMute: document.getElementById("btn-mute"),
    btnMode: document.getElementById("btn-mode"),
    btnPtt: document.getElementById("btn-ptt"),
    selectLanguage: document.getElementById("select-language"),
    selectDelegationMode: document.getElementById("select-delegation-mode"),
    chkFiller: document.getElementById("chk-filler"),
    selectPersona: document.getElementById("select-persona"),
    selectPatience: document.getElementById("select-patience"),
    chkTools: document.getElementById("chk-tools"),
    transcriptFeed: document.getElementById("transcript-feed"),
    textForm: document.getElementById("text-form"),
    textInput: document.getElementById("text-input"),
    btnExport: document.getElementById("btn-export"),
    btnExportMetrics: document.getElementById("btn-export-metrics"),
    statsBadge: document.getElementById("stats-badge"),
    statsMetaSub: document.getElementById("stats-meta-sub"),
    kpiPerceived: document.getElementById("kpi-perceived"),
    kpiPerceivedSub: document.getElementById("kpi-perceived-sub"),
    kpiServerTtfa: document.getElementById("kpi-server-ttfa"),
    kpiServerTtfaSub: document.getElementById("kpi-server-ttfa-sub"),
    kpiTtft: document.getElementById("kpi-ttft"),
    kpiTtftSub: document.getElementById("kpi-ttft-sub"),
    kpiBarge: document.getElementById("kpi-barge"),
    kpiBargeSub: document.getElementById("kpi-barge-sub"),
    kpiEndpoint: document.getElementById("kpi-endpoint"),
    kpiEndpointSub: document.getElementById("kpi-endpoint-sub"),
    kpiFiller: document.getElementById("kpi-filler"),
    kpiFillerSub: document.getElementById("kpi-filler-sub"),
    kpiDelegation: document.getElementById("kpi-delegation"),
    kpiDelegationSub: document.getElementById("kpi-delegation-sub"),
    kpiRtf: document.getElementById("kpi-rtf"),
    kpiRtfSub: document.getElementById("kpi-rtf-sub"),
    statsTurnsBody: document.getElementById("stats-turns-body"),
    delegationLog: document.getElementById("delegation-log"),
    flametreePortalWrap: document.getElementById("flametree-portal-wrap"),
    crmBadge: document.getElementById("crm-badge"),
    crmQualStatus: document.getElementById("crm-qual-status"),
    crmQualBody: document.getElementById("crm-qual-body"),
    crmCalcStatus: document.getElementById("crm-calc-status"),
    crmCalcBody: document.getElementById("crm-calc-body"),
    crmAppStatus: document.getElementById("crm-app-status"),
    crmAppBody: document.getElementById("crm-app-body"),
    crmCbStatus: document.getElementById("crm-cb-status"),
    crmCbBody: document.getElementById("crm-cb-body"),
    toolLog: document.getElementById("tool-log"),
    productSheetTitle: document.getElementById("product-sheet-title"),
    productComplianceNote: document.getElementById("product-compliance-note"),
    productTiers: document.getElementById("product-tiers"),
    orbCanvas: document.getElementById("orb-canvas"),
  };

  async function init() {
    bindTabs();
    bindControls();
    startOrbLoop();
    await loadConfig();
    connectWebSocket();
  }

  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      const cfg = await res.json();
      state.config = cfg;
      state.targetSec = cfg.target_call_duration_sec || 300;
      els.companyName.textContent = cfg.company_name;
      els.subtitle.textContent = cfg.subtitle;
      const backendTag = cfg.backend === "vllm" ? "vLLM-Omni" : "MLX";
      els.modelPill.textContent = `${cfg.model_id.split("/").pop()} · ${backendTag}`;
      if (els.pipelinePill) {
        els.pipelinePill.textContent =
          cfg.backend === "vllm"
            ? "Pure S2S · vLLM-Omni"
            : "Pure S2S · B=2 MLX";
      }

      els.badgeList.innerHTML = "";
      (cfg.badges || []).forEach((b) => {
        const span = document.createElement("span");
        span.className = "reg-badge";
        span.textContent = b;
        els.badgeList.appendChild(span);
      });

      if (cfg.default_language) {
        els.selectLanguage.value = cfg.default_language;
      }
      if (cfg.default_delegation_mode) {
        els.selectDelegationMode.value = cfg.default_delegation_mode;
      }
      if (typeof cfg.default_dynamic_filler === "boolean") {
        els.chkFiller.checked = cfg.default_dynamic_filler;
      }

      if (cfg.flametree) {
        renderFlametreePill(
          cfg.flametree.live_token_present,
          cfg.flametree.tenant_name,
          cfg.flametree.portal_url,
          null,
          null,
          null
        );
      }

      els.selectPersona.innerHTML = "";
      (cfg.personas || []).forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `${p.name} (${p.voice_gender})`;
        els.selectPersona.appendChild(opt);
      });
      if (cfg.default_persona_id) {
        els.selectPersona.value = cfg.default_persona_id;
      }
      updatePersonaDisplay();

      if (cfg.default_turn_patience) {
        els.selectPatience.value = cfg.default_turn_patience;
      }
      if (cfg.product_summary) {
        renderProductSheet(cfg.product_summary);
      }
    } catch (_) {
      els.subtitle.textContent = "Failed to load /api/config";
    }
  }

  function renderFlametreePill(isLive, tenantName, portalUrl, sessionId, errorMsg, sessionUrl) {
    if (errorMsg) {
      els.flametreePill.textContent = "Flametree: API Error";
      els.flametreePill.title = errorMsg;
      els.flametreePill.classList.remove("connected");
      els.flametreePill.classList.add("muted");
    } else if (isLive) {
      const sidTag = sessionId ? ` · ${String(sessionId).slice(0, 8)}` : "";
      els.flametreePill.textContent = `Flametree: Live (${tenantName || "prod"})${sidTag}`;
      els.flametreePill.title = sessionId ? `Active session: ${sessionId}` : "";
      els.flametreePill.classList.remove("muted");
      els.flametreePill.classList.add("connected");
    } else {
      els.flametreePill.textContent = "Flametree: Local KB Fallback";
      els.flametreePill.title = "FLAMETREE_AGENT_TOKEN unset; using offline fallback";
      els.flametreePill.classList.remove("connected");
      els.flametreePill.classList.add("muted");
    }
    const links = [];
    if (portalUrl) {
      links.push(`<a class="portal-link" href="${escapeHtml(portalUrl)}" target="_blank" rel="noopener noreferrer">Open Agent in Flametree Portal</a>`);
    }
    if (sessionUrl) {
      links.push(`<a class="portal-link" href="${escapeHtml(sessionUrl)}" target="_blank" rel="noopener noreferrer">Open Session</a>`);
    }
    if (errorMsg) {
      links.push(`<span class="context-sub" style="margin-left:8px;color:#ef5350">${escapeHtml(errorMsg)}</span>`);
    }
    els.flametreePortalWrap.innerHTML = links.join(" ");
  }

  function renderProductSheet(ps) {
    els.productSheetTitle.textContent = `${ps.product_name} — Tariff & Eligibility Sheet`;
    els.productComplianceNote.textContent = ps.compliance_note || "";
    els.productTiers.innerHTML = "";
    (ps.tiers || []).forEach((t) => {
      const box = document.createElement("div");
      box.className = "tier-box";
      box.innerHTML = `
        <h5>${escapeHtml(t.name)}</h5>
        <div><strong>Fee:</strong> ${escapeHtml(t.monthly_fee)}</div>
        <div><strong>Limit:</strong> ${escapeHtml(t.limit_range)}</div>
        <div><strong>Min Income:</strong> ${escapeHtml(t.min_income)}</div>
        <div><strong>Rate:</strong> ${escapeHtml(t.interest_rate)}</div>
        <div><strong>Cashback:</strong> ${escapeHtml(t.cashback)}</div>
        <ul>${(t.highlights || []).map((h) => `<li>${escapeHtml(h)}</li>`).join("")}</ul>
      `;
      els.productTiers.appendChild(box);
    });
  }

  function updatePersonaDisplay() {
    if (!state.config) return;
    const pid = els.selectPersona.value;
    const p = (state.config.personas || []).find((x) => x.id === pid);
    if (p) {
      els.personaName.textContent = p.name;
      els.personaRole.textContent = `${p.role} · ${els.selectLanguage.value}`;
    }
  }

  function connectWebSocket() {
    if (state.ws && state.ws.readyState <= WebSocket.OPEN) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/call`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      els.connPill.textContent = "Connected";
      els.connPill.classList.remove("muted");
      els.connPill.classList.add("connected");
    };

    ws.onclose = () => {
      els.connPill.textContent = "Disconnected";
      els.connPill.classList.remove("connected");
      els.connPill.classList.add("muted");
      stopCallMedia();
      setTimeout(connectWebSocket, 2000);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        handleIncomingAudio(event.data);
      } else if (typeof event.data === "string") {
        try {
          handleControlMessage(JSON.parse(event.data));
        } catch (err) {
          console.error("[CallStudio] Control message error:", err, event.data);
        }
      }
    };

    state.ws = ws;
  }

  function waitForWebSocket(timeoutMs = 4000) {
    return new Promise((resolve, reject) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      if (!state.ws || state.ws.readyState > WebSocket.OPEN) {
        connectWebSocket();
      }
      const start = Date.now();
      const timer = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
          clearInterval(timer);
          resolve();
        } else if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          reject(new Error("WebSocket connection timed out"));
        }
      }, 50);
    });
  }

  function sendJson(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  }

  function handleIncomingAudio(arrayBuffer) {
    stopRingtone();
    if (!state.acceptingAudio) {
      return;
    }
    if (state.playbackNode) {
      state.playbackNode.port.postMessage(
        { type: "audio", buffer: arrayBuffer },
        [arrayBuffer]
      );
    }
  }

  function handleControlMessage(msg) {
    switch (msg.type) {
      case "call_started":
        state.callId = msg.call_id || null;
        state.callArtifactsDir = msg.artifacts_dir || "";
        if (els.callIdReadout) {
          els.callIdReadout.textContent = msg.call_id
            ? `Call ID: ${msg.call_id}`
            : "Call ID: —";
          els.callIdReadout.title = msg.artifacts_dir || "";
        }
        break;

      case "call_state":
        setCallState(msg.state);
        break;

      case "delegation_status":
        renderFlametreePill(
          Boolean(msg.live),
          msg.tenant_name,
          state.config?.flametree?.portal_url,
          msg.session_id,
          msg.error,
          msg.session_url
        );
        break;

      case "s2s_listen_state":
        els.s2sIndicator.textContent = msg.speaking
          ? "S2S model hearing speech"
          : state.muted
          ? "Mic muted"
          : state.inputMode === "ptt"
          ? "Push-to-Talk ready"
          : `Streaming to S2S (${els.selectLanguage.value})`;
        break;

      case "barge_in":
        stopRingtone();
        state.acceptingAudio = false;
        state.agentRms = 0;
        if (state.playbackNode) {
          state.playbackNode.port.postMessage({ type: "clear" });
        }
        break;

      case "user_turn":
        ensureFeedStarted();
        addTranscriptMessage(msg.turn_id, "user", "You", msg.text, Boolean(msg.pending_asr));
        break;

      case "user_transcript_update":
        updateTranscriptMessage(msg.turn_id, msg.text, Boolean(msg.pending_asr));
        break;

      case "assistant_turn_start":
        ensureFeedStarted();
        state.acceptingAudio = true;
        if (state.playbackNode) {
          state.playbackNode.port.postMessage({ type: "new_turn", turnId: msg.turn_id });
        }
        addTranscriptMessage(
          msg.turn_id,
          "assistant",
          msg.persona_name || "Agent",
          "",
          false
        );
        break;

      case "text_delta":
        updateTranscriptMessage(msg.turn_id, msg.text, false);
        break;

      case "assistant_turn_end":
      case "assistant_turn_truncated":
        updateTranscriptMessage(msg.turn_id, msg.text, false);
        if (msg.metrics) {
          updateTranscriptMetrics(msg.turn_id, msg.metrics);
        }
        break;

      case "audio_end":
        if (state.playbackNode) {
          state.playbackNode.port.postMessage({ type: "end_of_stream" });
        }
        break;

      case "tool_event":
        ensureFeedStarted();
        addToolBubble(msg.tool, msg.arguments, msg.result);
        break;

      case "delegation_event":
        ensureFeedStarted();
        addDelegationTrace(msg);
        break;

      case "crm_state":
        renderCrmState(msg.state);
        break;

      case "metrics":
        els.teleTtfa.textContent = `${Math.round(msg.ttfa_ms)} ms`;
        els.teleSpeed.textContent = `${msg.slots_per_sec} slots/s`;
        if (msg.turn_id) {
          updateTranscriptMetrics(msg.turn_id, {
            ttft_ms: msg.ttft_ms,
            ttfa_server_ms: msg.ttfa_ms,
            route_decision: msg.route_decision,
          });
        }
        break;

      case "live_ttft":
        if (msg.ttft_ms != null) {
          els.kpiTtft.textContent = fmtMs(msg.ttft_ms);
          if (msg.turn_id) {
            updateTranscriptMetrics(msg.turn_id, { ttft_ms: msg.ttft_ms });
          }
        }
        break;

      case "live_ttfa":
        if (msg.ttfa_server_ms != null) {
          const val = fmtMs(msg.ttfa_server_ms);
          els.teleTtfa.textContent = val;
          els.kpiServerTtfa.textContent = val;
          if (msg.turn_id) {
            updateTranscriptMetrics(msg.turn_id, { ttfa_server_ms: msg.ttfa_server_ms });
          }
        }
        if (msg.ttfa_perceived_ms != null) {
          const val = fmtMs(msg.ttfa_perceived_ms);
          els.teleTtfaPerceived.textContent = val;
          els.kpiPerceived.textContent = val;
          if (msg.turn_id) {
            updateTranscriptMetrics(msg.turn_id, { ttfa_perceived_ms: msg.ttfa_perceived_ms });
          }
        }
        break;

      case "delivery_stats":
        state.latestDeliveryStats = msg.stats;
        renderDeliveryStats(msg.stats);
        break;

      case "context_stats": {
        const tokens = Number(msg.tokens || 0);
        const maxTok = Number(msg.max_tokens || 32768);
        const pct = Math.min(100, Math.round((tokens / maxTok) * 100));
        els.ctxBarFill.style.width = `${pct}%`;
        els.ctxReadout.textContent = `${tokens.toLocaleString()} / 32K · ${msg.compactions || 0} compactions`;
        els.listenerReadout.textContent = `LiveS2S Listener window: ${Number(msg.listener_sec || 0).toFixed(1)}s / 30.0s (${msg.listener_resets || 0} rollovers)`;
        break;
      }

      case "error":
        ensureFeedStarted();
        addToolBubble("system_error", {}, { error: msg.message });
        break;
    }
  }

  function fmtMs(v) {
    return v == null ? "—" : `${Math.round(Number(v))} ms`;
  }

  function fmtSubP(agg) {
    if (!agg || agg.count === 0) return "p50 — · p95 —";
    return `p50 ${fmtMs(agg.p50)} · p95 ${fmtMs(agg.p95)}`;
  }

  function renderDeliveryStats(stats) {
    if (!stats) return;
    const count = Number(stats.turn_count || 0);
    if (count > 0) {
      els.statsBadge.textContent = String(count);
      els.statsBadge.classList.remove("hidden");
    } else {
      els.statsBadge.classList.add("hidden");
    }

    const agg = stats.aggregates || {};
    if (agg.ttfa_perceived_ms?.last != null) {
      els.teleTtfaPerceived.textContent = fmtMs(agg.ttfa_perceived_ms.last);
    }
    if (agg.ttfa_server_ms?.last != null) {
      els.teleTtfa.textContent = fmtMs(agg.ttfa_server_ms.last);
    }
    if (agg.ttfa_after_barge_in_ms?.last != null) {
      els.teleBargeTtfa.textContent = fmtMs(agg.ttfa_after_barge_in_ms.last);
    } else {
      els.teleBargeTtfa.textContent = stats.barge_in_count > 0 ? "Pending" : "—";
    }
    if (agg.gen_rtf?.last != null && agg.slots_per_sec?.last != null) {
      els.teleSpeed.textContent = `${Number(agg.gen_rtf.last).toFixed(2)}x · ${Math.round(agg.slots_per_sec.last)}/s`;
    }

    els.kpiPerceived.textContent = fmtMs(agg.ttfa_perceived_ms?.last);
    els.kpiPerceivedSub.textContent = fmtSubP(agg.ttfa_perceived_ms);

    els.kpiServerTtfa.textContent = fmtMs(agg.ttfa_server_ms?.last);
    els.kpiServerTtfaSub.textContent = fmtSubP(agg.ttfa_server_ms);

    els.kpiTtft.textContent = fmtMs(agg.ttft_ms?.last);
    els.kpiTtftSub.textContent = fmtSubP(agg.ttft_ms);

    els.kpiBarge.textContent = fmtMs(agg.ttfa_after_barge_in_ms?.last);
    els.kpiBargeSub.textContent = `Cancel avg ${fmtMs(agg.barge_in_cancel_ms?.avg)} · ${stats.barge_in_count || 0} barge-ins`;

    els.kpiEndpoint.textContent = fmtMs(agg.endpoint_ms?.last);
    els.kpiEndpointSub.textContent = fmtSubP(agg.endpoint_ms);

    els.kpiFiller.textContent = fmtMs(agg.ttfa_filler_ms?.last);
    els.kpiFillerSub.textContent = fmtSubP(agg.ttfa_filler_ms);

    els.kpiDelegation.textContent = fmtMs(agg.delegation_ms?.last);
    els.kpiDelegationSub.textContent = fmtSubP(agg.delegation_ms);

    const lastRtf = agg.gen_rtf?.last;
    els.kpiRtf.textContent = lastRtf == null ? "—" : `${Number(lastRtf).toFixed(2)}x`;
    const avgCodec = agg.codec_chunk_avg_ms?.avg;
    els.kpiRtfSub.textContent = `Codec chunk avg ${avgCodec == null ? "—" : `${Number(avgCodec).toFixed(1)} ms`}`;

    const rows = stats.recent_turns || [];
    if (rows.length === 0) {
      els.statsTurnsBody.innerHTML = '<tr><td colspan="10" class="muted">No turns recorded yet.</td></tr>';
      return;
    }
    els.statsTurnsBody.innerHTML = rows
      .slice()
      .reverse()
      .map(
        (t) => `<tr>
          <td>#${t.turn_index}</td>
          <td>${escapeHtml(t.language)}</td>
          <td><span class="route-pill ${escapeHtml(t.route_decision)}">${escapeHtml(t.route_decision)}${t.after_barge_in ? " ·bi" : ""}${t.interrupted ? " ·int" : ""}</span></td>
          <td>${t.audio_chunks_sent}</td>
          <td>${fmtMs(t.endpoint_ms)}</td>
          <td>${fmtMs(t.ttft_ms)}</td>
          <td>${fmtMs(t.ttfa_server_ms)}</td>
          <td>${fmtMs(t.ttfa_perceived_ms)}</td>
          <td>${fmtMs(t.delegation_ms)}</td>
          <td>${Number(t.gen_rtf || 0).toFixed(2)}x</td>
        </tr>`
      )
      .join("");
  }

  function addDelegationTrace(ev) {
    state.delegationLog.unshift(ev);
    const row = document.createElement("div");
    row.className = "msg-row delegation";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = `Flametree [${ev.source} · ${Math.round(ev.latency_ms)} ms]: "${ev.query}"`;
    row.appendChild(bubble);
    els.transcriptFeed.appendChild(row);
    els.transcriptFeed.scrollTop = els.transcriptFeed.scrollHeight;

    els.delegationLog.classList.remove("muted");
    els.delegationLog.innerHTML = state.delegationLog
      .map(
        (d) => `<div class="tool-log-item">
          <strong>[${escapeHtml(d.language)} · ${escapeHtml(d.source)} · ${Math.round(d.latency_ms)} ms]</strong><br/>
          Query: <code>${escapeHtml(d.query)}</code><br/>
          Answer: ${escapeHtml(d.answer)}${d.error ? `<br/><span class="muted">Note: ${escapeHtml(d.error)}</span>` : ""}
        </div>`
      )
      .join("");
  }

  function setCallState(nextState) {
    state.callState = nextState;
    const labels = {
      idle: "Ready to Dial",
      ringing: "Ringing…",
      listening: "Streaming to S2S",
      thinking: "Generating…",
      speaking: "Agent Speaking",
    };
    els.callStateLabel.textContent = labels[nextState] || nextState;
    if (nextState === "ringing") {
      startRingtone();
    } else {
      stopRingtone();
    }
    if (nextState === "idle") {
      state.agentRms = 0;
    }
  }

  async function ensureAudioPipeline() {
    if (!state.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AudioCtx();
    }
    if (state.audioCtx.state === "suspended") {
      await state.audioCtx.resume();
    }

    if (!state.playbackNode) {
      await state.audioCtx.audioWorklet.addModule("/static/playback-worklet.js?v=20261122-3");
      state.playbackNode = new AudioWorkletNode(state.audioCtx, "playback-worklet");
      state.playbackNode.connect(state.audioCtx.destination);
      state.playbackNode.port.onmessage = (event) => {
        const d = event.data;
        if (!d) return;
        if (d.type === "started") {
          sendJson({
            type: "playback_started",
            turn_id: d.turnId,
            buffered_samples: d.bufferedSamples,
          });
        } else if (d.type === "progress") {
          state.agentRms = d.rms || 0;
          sendJson({ type: "playback_progress", played_samples: d.playedSamples });
        } else if (d.type === "drained") {
          state.agentRms = 0;
          sendJson({ type: "playback_done", played_samples: d.playedSamples });
        }
      };
    }

    if (!state.captureNode && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        state.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        await state.audioCtx.audioWorklet.addModule("/static/capture-worklet.js?v=20261122-3");
        const source = state.audioCtx.createMediaStreamSource(state.micStream);
        state.captureNode = new AudioWorkletNode(state.audioCtx, "capture-worklet");
        const silentGain = state.audioCtx.createGain();
        silentGain.gain.value = 0.0;
        source.connect(state.captureNode);
        state.captureNode.connect(silentGain);
        silentGain.connect(state.audioCtx.destination);

        state.captureNode.port.onmessage = (event) => {
          const d = event.data;
          if (!d || d.type !== "pcm16") return;
          state.micRms = state.muted ? 0 : d.rms || 0;
          if (state.callActive && !state.muted && state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(d.buffer);
          }
        };
      } catch (_) {
        els.s2sIndicator.textContent = "Mic blocked — use text input";
      }
    }
  }

  async function startCall() {
    await ensureAudioPipeline();
    try {
      await waitForWebSocket();
    } catch (_) {
      els.s2sIndicator.textContent = "Server offline — retrying…";
      return;
    }
    state.callActive = true;
    state.muted = false;
    state.callId = null;
    state.callArtifactsDir = "";
    if (els.callIdReadout) {
      els.callIdReadout.textContent = "Call ID: —";
      els.callIdReadout.title = "";
    }
    state.turns.clear();
    state.pendingMetrics.clear();
    state.transcriptLog = [];
    state.delegationLog = [];
    els.transcriptFeed.innerHTML = "";
    els.delegationLog.className = "tool-log muted";
    els.delegationLog.textContent =
      "Knowledge delegation queries extracted in parallel via MLX B=2 appear here.";
    els.btnCall.textContent = "Hang Up";
    els.btnCall.classList.add("active-call");
    els.btnMute.disabled = false;
    els.btnMute.textContent = "Mute Mic";
    els.btnMute.classList.remove("active-toggle");
    els.btnPtt.disabled = false;
    els.s2sIndicator.textContent = `Streaming to S2S (${els.selectLanguage.value})`;

    state.callStartTs = Date.now();
    clearInterval(state.timerInterval);
    state.timerInterval = setInterval(updateTimer, 1000);
    updateTimer();

    sendJson({
      type: "start_call",
      persona_id: els.selectPersona.value,
      tools_enabled: els.chkTools.checked,
      turn_patience: els.selectPatience.value,
      input_mode: state.inputMode,
      language: els.selectLanguage.value,
      delegation_mode: els.selectDelegationMode.value,
      dynamic_filler: els.chkFiller.checked,
    });
  }

  function hangUp() {
    sendJson({ type: "hang_up" });
    stopCallMedia();
  }

  function stopCallMedia() {
    state.callActive = false;
    state.acceptingAudio = false;
    stopRingtone();
    clearInterval(state.timerInterval);
    state.agentRms = 0;
    state.micRms = 0;
    if (state.playbackNode) {
      state.playbackNode.port.postMessage({ type: "clear" });
    }
    els.btnCall.textContent = "Start Call";
    els.btnCall.classList.remove("active-call");
    els.btnMute.disabled = true;
    els.btnPtt.disabled = true;
    els.s2sIndicator.textContent = "S2S stream idle";
    setCallState("idle");
  }

  function fmtMMSS(totalSec) {
    const mm = String(Math.floor(totalSec / 60)).padStart(2, "0");
    const ss = String(totalSec % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }

  function updateTimer() {
    if (!state.callActive) return;
    const sec = Math.floor((Date.now() - state.callStartTs) / 1000);
    els.callTimer.textContent = `${fmtMMSS(sec)} / ${fmtMMSS(state.targetSec)}`;
  }

  function startRingtone() {
    if (state.ringInterval || !state.audioCtx) return;
    const playBurst = (delaySec, durSec) => {
      const ctx = state.audioCtx;
      const now = ctx.currentTime + delaySec;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 400;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.05, now + 0.02);
      gain.gain.setValueAtTime(0.05, now + durSec - 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + durSec);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + durSec + 0.02);
    };
    const playCadence = () => {
      if (!state.audioCtx || state.callState !== "ringing") return;
      playBurst(0.0, 0.38);
      playBurst(0.58, 0.38);
    };
    playCadence();
    state.ringInterval = setInterval(playCadence, 2800);
  }

  function stopRingtone() {
    if (state.ringInterval) {
      clearInterval(state.ringInterval);
      state.ringInterval = null;
    }
  }

  function ensureFeedStarted() {
    const empty = els.transcriptFeed.querySelector(".empty-state");
    if (empty) empty.remove();
  }

  function addTranscriptMessage(turnId, role, speaker, text, pending) {
    if (state.turns.has(turnId)) {
      updateTranscriptMessage(turnId, text, pending);
      return;
    }
    const row = document.createElement("div");
    row.className = `msg-row ${role}${pending ? " pending" : ""}`;
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = `${speaker} · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = text || "…";
    row.appendChild(meta);
    row.appendChild(bubble);

    let metricsEl = null;
    if (role === "assistant") {
      metricsEl = document.createElement("div");
      metricsEl.className = "msg-metrics hidden";
      row.appendChild(metricsEl);
    }

    els.transcriptFeed.appendChild(row);
    els.transcriptFeed.scrollTop = els.transcriptFeed.scrollHeight;

    const record = {
      rowEl: row,
      bubbleEl: bubble,
      metricsEl,
      role,
      speaker,
      text: text || "",
      metrics: {},
    };
    state.turns.set(turnId, record);
    state.transcriptLog.push(record);
    if (role === "assistant" && state.pendingMetrics.has(turnId)) {
      const earlyPatch = state.pendingMetrics.get(turnId);
      state.pendingMetrics.delete(turnId);
      updateTranscriptMetrics(turnId, earlyPatch);
    }
  }

  function formatTranscriptMetricsList(m) {
    if (!m) return [];
    const parts = [];
    if (m.ttft_ms != null) parts.push(`TTFT: ${Math.round(Number(m.ttft_ms))} ms`);
    if (m.ttfa_server_ms != null) parts.push(`TTFA: ${Math.round(Number(m.ttfa_server_ms))} ms`);
    if (m.ttfa_perceived_ms != null) {
      parts.push(`Perceived: ${Math.round(Number(m.ttfa_perceived_ms))} ms`);
    }
    return parts;
  }

  function updateTranscriptMetrics(turnId, patch) {
    if (!turnId || !patch) return;
    const entry = state.turns.get(turnId);
    if (!entry) {
      const prev = state.pendingMetrics.get(turnId) || {};
      Object.entries(patch).forEach(([k, v]) => {
        if (v != null) prev[k] = v;
      });
      state.pendingMetrics.set(turnId, prev);
      return;
    }
    Object.entries(patch).forEach(([k, v]) => {
      if (v != null) entry.metrics[k] = v;
    });
    if (!entry.metricsEl) return;
    const parts = formatTranscriptMetricsList(entry.metrics);
    if (parts.length === 0) return;
    entry.metricsEl.classList.remove("hidden");
    entry.metricsEl.innerHTML = parts
      .map((p) => `<span class="msg-metric-chip">${escapeHtml(p)}</span>`)
      .join("");
  }

  function updateTranscriptMessage(turnId, text, pending) {
    const entry = state.turns.get(turnId);
    if (!entry) return;
    entry.text = text || "";
    entry.bubbleEl.textContent = entry.text || "…";
    if (!pending) {
      entry.rowEl.classList.remove("pending");
    }
    els.transcriptFeed.scrollTop = els.transcriptFeed.scrollHeight;
  }

  function addToolBubble(toolName, args, result) {
    const row = document.createElement("div");
    row.className = "msg-row tool";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = `CRM Tool: ${toolName}(${JSON.stringify(args)}) -> ${result?.summary || JSON.stringify(result)}`;
    row.appendChild(bubble);
    els.transcriptFeed.appendChild(row);
    els.transcriptFeed.scrollTop = els.transcriptFeed.scrollHeight;
  }

  function renderCrmState(crm) {
    if (!crm) return;
    const count = (crm.tool_log || []).length;
    state.crmToolCount = count;
    if (count > 0) {
      els.crmBadge.textContent = String(count);
      els.crmBadge.classList.remove("hidden");
    } else {
      els.crmBadge.classList.add("hidden");
    }

    if (crm.qualification) {
      const q = crm.qualification;
      els.crmQualStatus.textContent = q.eligible ? q.recommended_tier : "Declined";
      els.crmQualStatus.className = `crm-tag ${q.eligible ? "done" : ""}`;
      els.crmQualBody.classList.remove("muted");
      els.crmQualBody.innerHTML = q.eligible
        ? `<strong>Tier:</strong> ${escapeHtml(q.recommended_tier)}<br/>
           <strong>Est. Limit:</strong> R${Number(q.estimated_limit_zar).toLocaleString()}<br/>
           <strong>Rate:</strong> ${q.indicative_rate_pct}% p.a. · Fee R${q.monthly_fee_zar}/mo`
        : escapeHtml(q.reason || "Did not meet criteria.");
    } else {
      els.crmQualStatus.textContent = "Pending";
      els.crmQualStatus.className = "crm-tag";
      els.crmQualBody.className = "crm-card-body muted";
      els.crmQualBody.textContent = "No pre-qualification check run yet.";
    }

    if (crm.repayment) {
      const r = crm.repayment;
      els.crmCalcStatus.textContent = `R${Number(r.monthly_repayment_zar).toLocaleString()}/mo`;
      els.crmCalcStatus.className = "crm-tag done";
      els.crmCalcBody.classList.remove("muted");
      els.crmCalcBody.innerHTML = `<strong>Amount:</strong> R${Number(r.principal_zar).toLocaleString()}<br/>
        <strong>Term:</strong> ${r.months} months @ ${r.annual_rate_pct}%<br/>
        <strong>Total Interest:</strong> R${Number(r.total_interest_zar).toLocaleString()}`;
    } else {
      els.crmCalcStatus.textContent = "Pending";
      els.crmCalcStatus.className = "crm-tag";
      els.crmCalcBody.className = "crm-card-body muted";
      els.crmCalcBody.textContent = "No repayment quote calculated yet.";
    }

    if (crm.application) {
      const a = crm.application;
      els.crmAppStatus.textContent = a.reference_number;
      els.crmAppStatus.className = "crm-tag done";
      els.crmAppBody.classList.remove("muted");
      els.crmAppBody.innerHTML = `<strong>Applicant:</strong> ${escapeHtml(a.applicant_name)}<br/>
        <strong>Tier:</strong> ${escapeHtml(a.card_tier)}<br/>
        <strong>Status:</strong> ${escapeHtml(a.status)} (POPIA Consented)`;
    } else {
      els.crmAppStatus.textContent = "Pending";
      els.crmAppStatus.className = "crm-tag";
      els.crmAppBody.className = "crm-card-body muted";
      els.crmAppBody.textContent = "POPIA consent & provisional FICA application not submitted.";
    }

    if (crm.callback) {
      const c = crm.callback;
      els.crmCbStatus.textContent = "Booked";
      els.crmCbStatus.className = "crm-tag done";
      els.crmCbBody.classList.remove("muted");
      els.crmCbBody.innerHTML = `<strong>When:</strong> ${escapeHtml(c.preferred_time)}<br/>
        <strong>Phone:</strong> ${escapeHtml(c.phone_number)}<br/>
        <strong>Ref:</strong> ${escapeHtml(c.callback_id)}`;
    } else {
      els.crmCbStatus.textContent = "None";
      els.crmCbStatus.className = "crm-tag";
      els.crmCbBody.className = "crm-card-body muted";
      els.crmCbBody.textContent = "No follow-up callback scheduled.";
    }

    if (count > 0) {
      els.toolLog.classList.remove("muted");
      els.toolLog.innerHTML = (crm.tool_log || [])
        .map(
          (item) => `<div class="tool-log-item">
            <strong>[${escapeHtml(item.timestamp)}] ${escapeHtml(item.tool)}</strong><br/>
            Args: <code>${escapeHtml(JSON.stringify(item.arguments))}</code><br/>
            Result: <code>${escapeHtml(JSON.stringify(item.result))}</code>
          </div>`
        )
        .join("");
    } else {
      els.toolLog.className = "tool-log muted";
      els.toolLog.textContent = "Tool calls executed by LFM3-Omni-2B during the call appear here live.";
    }
  }

  function bindTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const target = document.getElementById(`tab-${btn.dataset.tab}`);
        if (target) target.classList.add("active");
      });
    });
  }

  function bindControls() {
    els.btnCall.addEventListener("click", () => {
      if (state.callActive) hangUp();
      else startCall();
    });

    els.btnMute.addEventListener("click", () => {
      state.muted = !state.muted;
      els.btnMute.textContent = state.muted ? "Unmute Mic" : "Mute Mic";
      els.btnMute.classList.toggle("active-toggle", state.muted);
      sendJson({ type: "set_config", muted: state.muted });
    });

    els.btnMode.addEventListener("click", () => {
      state.inputMode = state.inputMode === "live" ? "ptt" : "live";
      const isPtt = state.inputMode === "ptt";
      els.btnMode.textContent = isPtt ? "Mode: Push-to-Talk" : "Mode: Live S2S";
      els.btnMode.classList.toggle("active-toggle", isPtt);
      els.btnPtt.classList.toggle("hidden", !isPtt);
      sendJson({ type: "set_config", input_mode: state.inputMode });
    });

    const pttStart = () => {
      if (!state.callActive || state.inputMode !== "ptt") return;
      els.btnPtt.classList.add("holding");
      sendJson({ type: "ptt_start" });
    };
    const pttEnd = () => {
      if (!els.btnPtt.classList.contains("holding")) return;
      els.btnPtt.classList.remove("holding");
      sendJson({ type: "ptt_end" });
    };
    els.btnPtt.addEventListener("mousedown", pttStart);
    window.addEventListener("mouseup", pttEnd);
    els.btnPtt.addEventListener("touchstart", (e) => {
      e.preventDefault();
      pttStart();
    });
    window.addEventListener("touchend", pttEnd);

    window.addEventListener("keydown", (e) => {
      if (
        e.code === "Space" &&
        state.inputMode === "ptt" &&
        state.callActive &&
        document.activeElement !== els.textInput &&
        !e.repeat
      ) {
        e.preventDefault();
        pttStart();
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space" && state.inputMode === "ptt") {
        pttEnd();
      }
    });

    els.selectLanguage.addEventListener("change", () => {
      updatePersonaDisplay();
      sendJson({ type: "set_config", language: els.selectLanguage.value });
    });

    els.selectDelegationMode.addEventListener("change", () => {
      sendJson({
        type: "set_config",
        delegation_mode: els.selectDelegationMode.value,
      });
    });

    els.chkFiller.addEventListener("change", () => {
      sendJson({ type: "set_config", dynamic_filler: els.chkFiller.checked });
    });

    els.selectPersona.addEventListener("change", () => {
      updatePersonaDisplay();
      sendJson({ type: "set_config", persona_id: els.selectPersona.value });
    });

    els.selectPatience.addEventListener("change", () => {
      sendJson({ type: "set_config", turn_patience: els.selectPatience.value });
    });

    els.chkTools.addEventListener("change", () => {
      sendJson({ type: "set_config", tools_enabled: els.chkTools.checked });
    });

    els.textForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = els.textInput.value.trim();
      if (!text) return;
      els.textInput.value = "";
      await ensureAudioPipeline();
      try {
        await waitForWebSocket();
      } catch (_) {
        els.s2sIndicator.textContent = "Server offline — retrying…";
        return;
      }
      if (!state.callActive) {
        state.callActive = true;
        state.callId = null;
        state.callArtifactsDir = "";
        if (els.callIdReadout) {
          els.callIdReadout.textContent = "Call ID: —";
          els.callIdReadout.title = "";
        }
        els.btnCall.textContent = "Hang Up";
        els.btnCall.classList.add("active-call");
        els.btnMute.disabled = false;
        els.btnPtt.disabled = false;
        state.callStartTs = Date.now();
        clearInterval(state.timerInterval);
        state.timerInterval = setInterval(updateTimer, 1000);
        updateTimer();
        sendJson({
          type: "set_config",
          language: els.selectLanguage.value,
          delegation_mode: els.selectDelegationMode.value,
          dynamic_filler: els.chkFiller.checked,
          persona_id: els.selectPersona.value,
          turn_patience: els.selectPatience.value,
          tools_enabled: els.chkTools.checked,
        });
      }
      sendJson({ type: "user_text", text });
    });

    els.btnExport.addEventListener("click", () => {
      const header = state.callId ? [`Call ID: ${state.callId}`] : [];
      const lines = header.concat(
        state.transcriptLog.map((entry) => {
          const mParts =
            entry.role === "assistant" ? formatTranscriptMetricsList(entry.metrics) : [];
          const mTag = mParts.length ? ` (${mParts.join(" | ")})` : "";
          return `[${entry.speaker}${mTag}] ${entry.text}`;
        })
      );
      const blob = new Blob([lines.join("\n\n")], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `karoo-crest-${state.callId || Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    });

    els.btnExportMetrics.addEventListener("click", () => {
      const payload = {
        call_id: state.callId,
        artifacts_dir: state.callArtifactsDir || null,
        exported_at: new Date().toISOString(),
        language: els.selectLanguage.value,
        delegation_mode: els.selectDelegationMode.value,
        dynamic_filler: els.chkFiller.checked,
        stats: state.latestDeliveryStats,
        delegation_log: state.delegationLog,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `karoo-crest-delivery-stats-${state.callId || Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function startOrbLoop() {
    const ctx = els.orbCanvas.getContext("2d");
    const w = els.orbCanvas.width;
    const h = els.orbCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    let phase = 0;

    const palette = {
      idle: [147, 164, 184],
      ringing: [245, 185, 66],
      listening: [78, 161, 255],
      thinking: [167, 139, 250],
      speaking: [24, 184, 138],
    };

    function draw() {
      phase += 0.04;
      ctx.clearRect(0, 0, w, h);

      const rgb = palette[state.callState] || palette.idle;
      const energy =
        state.callState === "speaking"
          ? Math.min(1, state.agentRms * 5.0)
          : state.callState === "listening"
          ? Math.min(1, state.micRms * 7.0)
          : state.callState === "ringing"
          ? 0.35 + 0.25 * Math.sin(phase * 3)
          : state.callState === "thinking"
          ? 0.25
          : 0.05;

      const baseR = 78 + energy * 16;

      const grad = ctx.createRadialGradient(cx, cy, baseR * 0.4, cx, cy, baseR + 46);
      grad.addColorStop(0, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.28 + energy * 0.25})`);
      grad.addColorStop(1, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR + 46, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.9)`;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const points = 72;
      for (let i = 0; i <= points; i++) {
        const theta = (i / points) * Math.PI * 2;
        const wobble =
          Math.sin(theta * 4 + phase * 1.8) * (2.5 + energy * 11) +
          Math.cos(theta * 7 - phase * 1.3) * (1.5 + energy * 6);
        const r = baseR + wobble;
        const x = cx + Math.cos(theta) * r;
        const y = cy + Math.sin(theta) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();

      ctx.fillStyle = `rgba(18, 26, 36, 0.85)`;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR - 10, 0, Math.PI * 2);
      ctx.fill();

      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  init();
})();
