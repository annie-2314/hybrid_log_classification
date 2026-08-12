"""Streamlit frontend for HybridLog Classifier.

Run:
    streamlit run ui/app.py

Requires the FastAPI backend unless Direct mode is selected:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.api_client import IntelliLogAPIError, IntelliLogClient  # noqa: E402

EXAMPLE_LOGS = {
    "Regex — backup": "Backup completed successfully.",
    "Regex — login": "User User123 logged in.",
    "BERT — security": "Unauthorized access to data was attempted",
    "BERT — HTTP": "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400",
    "LLM-style — workflow": "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
}

SOURCES = ["(none)", "ModernCRM", "ModernHR", "BillingSystem", "AnalyticsEngine", "ThirdPartyAPI", "LegacyCRM"]

SEVERITY_COLORS = {
    "CRITICAL": "#7f1d1d",
    "HIGH": "#b91c1c",
    "MEDIUM": "#b45309",
    "LOW": "#15803d",
    "UNKNOWN": "#475569",
}


def _direct_classify(log_message: str, source: Optional[str]) -> Dict[str, Any]:
    from app.routing.router import get_router

    return get_router().classify(log_message, source=source or None).model_dump()


def _direct_health() -> Dict[str, Any]:
    from app.core.config import get_settings
    from app.routing.router import get_router

    settings = get_settings()
    hybrid = get_router()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "bert_loaded": hybrid.bert.is_available,
        "legacy_st_lr_loaded": hybrid.legacy.is_available,
        "llm_configured": hybrid.llm.is_configured,
        "mode": "direct",
    }


def _direct_metrics() -> Dict[str, Any]:
    from app.monitoring.metrics import get_metrics_collector

    return get_metrics_collector().snapshot()


def _on_streamlit_cloud() -> bool:
    return bool(os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or os.getenv("HOSTNAME", "").endswith(".streamlit.app"))


def _api_available(client: IntelliLogClient) -> bool:
    try:
        client.health()
        return True
    except IntelliLogAPIError:
        return False


def classify_one(log_message: str, source: Optional[str], use_api: bool, client: IntelliLogClient) -> Dict[str, Any]:
    if use_api:
        try:
            return client.predict(log_message, source=source or None)
        except IntelliLogAPIError:
            return _direct_classify(log_message, source)
    return _direct_classify(log_message, source)


def render_result(result: Dict[str, Any]) -> None:
    severity = str(result.get("severity", "UNKNOWN")).upper()
    color = SEVERITY_COLORS.get(severity, "#334155")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:12px;background:{color}22;border:1px solid {color};">
            <div style="font-size:0.85rem;opacity:0.8;">Predicted category</div>
            <div style="font-size:1.6rem;font-weight:700;">{result.get("category", "-")}</div>
            <div style="margin-top:0.4rem;">Severity: <b>{severity}</b> · Method: <b>{result.get("classification_method", "-")}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{float(result.get('confidence') or 0):.3f}")
    c2.metric("Latency (ms)", f"{float(result.get('latency_ms') or 0):.1f}")
    c3.metric("LLM invoked", "Yes" if result.get("llm_invoked") else "No")
    st.write("**Explanation**")
    st.info(result.get("explanation") or "No explanation returned.")
    if result.get("routing_path"):
        st.caption("Routing path: " + " → ".join(result["routing_path"]))
    if result.get("request_id"):
        st.caption(f"Request ID: {result['request_id']}")


def main() -> None:
    st.set_page_config(page_title="HybridLog Classifier", page_icon="🪵", layout="wide")
    st.title("HybridLog Classifier")
    st.caption("Hybrid log classification: Regex → Fine-tuned BERT → LLM fallback")

    default_api = os.getenv("INTELLILOG_API_URL", "http://127.0.0.1:8000")
    with st.sidebar:
        st.header("Status")
        api_url = os.getenv("INTELLILOG_API_URL", default_api)
        client = IntelliLogClient(base_url=api_url)
        # Streamlit Cloud has no local FastAPI; use in-process router quietly.
        use_api = (not _on_streamlit_cloud()) and _api_available(client)
        health = _direct_health() if not use_api else client.health()
        st.success("Ready")
        st.write(f"BERT loaded: `{health.get('bert_loaded')}`")
        st.write(f"Legacy ST+LR: `{health.get('legacy_st_lr_loaded')}`")
        st.write(f"LLM configured: `{health.get('llm_configured')}`")
        st.caption("Engine: hybrid router (Regex + BERT + LLM)")

    tab_single, tab_batch, tab_metrics = st.tabs(["Classify log", "Batch CSV", "Monitoring"])

    with tab_single:
        st.subheader("Single log prediction")
        st.caption("Pick an example or paste a log, then click **Classify**. Results appear below the button.")

        if "log_message" not in st.session_state:
            st.session_state.log_message = EXAMPLE_LOGS["BERT — security"]

        def _apply_example() -> None:
            selected = st.session_state.example_key
            if selected != "(type your own)":
                st.session_state.log_message = EXAMPLE_LOGS[selected]

        st.selectbox(
            "Load example",
            ["(type your own)"] + list(EXAMPLE_LOGS.keys()),
            index=3,
            key="example_key",
            on_change=_apply_example,
        )
        log_message = st.text_area(
            "Log message",
            key="log_message",
            height=140,
            placeholder="Paste a log here, or choose an example above.",
        )
        source = st.selectbox("Source (optional)", SOURCES, index=0)
        source_value = None if source == "(none)" else source
        if st.button("Classify", type="primary"):
            if not str(log_message).strip():
                st.warning("Enter a log message first (or pick an example).")
            else:
                try:
                    with st.spinner("Classifying..."):
                        result = classify_one(str(log_message).strip(), source_value, use_api, client)
                    render_result(result)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Classification failed: {exc}")

    with tab_batch:
        st.subheader("Batch classification")
        st.caption("CSV must include a `log_message` column. Optional: `source`.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(df.head(20), use_container_width=True)
            if "log_message" not in df.columns:
                st.error("CSV must contain a `log_message` column.")
            elif st.button("Classify CSV", type="primary"):
                try:
                    logs: List[Dict[str, Any]] = []
                    for _, row in df.iterrows():
                        item: Dict[str, Any] = {"log_message": str(row["log_message"])}
                        if "source" in df.columns and pd.notna(row["source"]):
                            item["source"] = str(row["source"])
                        logs.append(item)
                    with st.spinner(f"Classifying {len(logs)} logs..."):
                        from app.services.batch import classify_dataframe
                        from app.routing.router import get_router

                        if use_api:
                            try:
                                payload = client.batch_predict(logs)
                                out_df = pd.DataFrame(payload.get("results", []))
                                if not out_df.empty:
                                    out_df = pd.concat([df.reset_index(drop=True), out_df], axis=1)
                            except IntelliLogAPIError:
                                out_df = classify_dataframe(df, router=get_router())
                        else:
                            out_df = classify_dataframe(df, router=get_router())
                    st.success(f"Classified {len(out_df)} rows.")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(
                        "Download results CSV",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name="classified_logs.csv",
                        mime="text/csv",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Batch classification failed: {exc}")

    with tab_metrics:
        st.subheader("Runtime monitoring")
        try:
            metrics = client.metrics() if use_api else _direct_metrics()
        except Exception:
            metrics = _direct_metrics()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total requests", metrics.get("total_requests", 0))
        m2.metric("Avg latency (ms)", f"{float(metrics.get('average_latency_ms') or 0):.2f}")
        m3.metric("LLM fallback %", f"{float(metrics.get('llm_fallback_percentage') or 0):.2f}")
        m4.metric("Est. LLM cost USD", f"{float(metrics.get('estimated_llm_cost_usd') or 0):.6f}")
        chart_df = pd.DataFrame(
            {
                "method": ["Regex", "BERT", "Legacy ST+LR", "LLM fallback"],
                "count": [
                    metrics.get("regex_requests", 0),
                    metrics.get("bert_requests", 0),
                    metrics.get("legacy_st_lr_requests", 0),
                    metrics.get("llm_fallback_requests", 0),
                ],
            }
        )
        st.bar_chart(chart_df.set_index("method"))
        st.json(metrics)


if __name__ == "__main__":
    main()
