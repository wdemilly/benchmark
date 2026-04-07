import io
import json
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

APP_TITLE = "Benchmark Harness"
DEFAULT_PROMPT = (
    "Read the attached benchmark packet from beginning to end.\n\n"
    "The packet contains the source passages that define the prose register, "
    "the benchmark scene outline, and the constraints for the piece.\n\n"
    "Write the benchmark scene from that outline in the prose register established "
    "by the source passages. Follow the packet’s constraints and target length. "
    "Return only the scene as plain text."
)

DATA_DIR = Path("benchmark_runs")
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    provider: str
    model: str
    prompt: str
    packet_name: Optional[str]
    packet_sha256: Optional[str]
    packet_mode: str
    max_tokens: int
    temperature: float
    notes: str
    output_file: str
    packet_file: str = ""
    payload_file: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_records(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "provider",
            "model",
            "prompt",
            "packet_name",
            "packet_sha256",
            "packet_mode",
            "max_tokens",
            "temperature",
            "notes",
            "output_file",
            "packet_file",
            "payload_file",
            "originality_label",
            "originality_score",
            "manual_rating",
            "manual_notes",
        ]
    )


def append_record(csv_path: Path, record: RunRecord) -> None:
    df = load_records(csv_path)
    df = pd.concat([df, pd.DataFrame([asdict(record)])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def update_record(csv_path: Path, run_id: str, updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty:
        return
    mask = df["run_id"] == run_id
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


def normalize_anthropic_text(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def normalize_packet_text(text: str) -> str:
    return (
        text.replace("\u2013", "-")    # en dash
            .replace("\u2014", "--")  # em dash
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u00a0", " ")
    )


def decode_packet_bytes(packet_bytes: Optional[bytes]) -> str:
    if not packet_bytes:
        return ""

    try:
        text = packet_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = packet_bytes.decode("latin-1")
        except Exception as exc:
            raise RuntimeError(
                "Packet file could not be decoded to text. Use a plain-text .txt or paste the packet text directly."
            ) from exc

    return normalize_packet_text(text)


def build_full_payload(prompt: str, packet_text: str, packet_name: Optional[str]) -> str:
    clean_prompt = prompt.strip()
    clean_packet = packet_text.strip()

    if not clean_prompt:
        raise RuntimeError("Prompt cannot be empty.")

    if not clean_packet:
        raise RuntimeError("Packet text is empty. The benchmark packet is required.")

    display_name = packet_name or "benchmark_packet.txt"

    return (
        f"BENCHMARK PACKET NAME: {display_name}\n"
        f"BEGIN BENCHMARK PACKET\n"
        f"{clean_packet}\n"
        f"END BENCHMARK PACKET\n\n"
        f"INSTRUCTION\n"
        f"{clean_prompt}\n"
        f"END INSTRUCTION"
    )


def call_anthropic(
    api_key: str,
    model: str,
    full_payload: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    client = anthropic.Anthropic(api_key=api_key)

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": full_payload,
            }
        ],
    )
    return normalize_anthropic_text(resp)


def call_openai(
    api_key: str,
    model: str,
    full_payload: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=full_payload,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return (getattr(resp, "output_text", "") or "").strip()


def export_zip(df: pd.DataFrame, outputs_root: Path) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(outputs_root.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(outputs_root.parent)))
    mem.seek(0)
    return mem.read()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Run repeated benchmark generations and log results for comparison against the web GUI.")

    csv_path = DATA_DIR / "runs.csv"
    outputs_dir = DATA_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    with st.sidebar:
        st.header("Run setup")
        provider = st.selectbox("Provider", ["anthropic", "openai"], index=0)
        model = st.text_input(
            "Model",
            value=("claude-sonnet-4-5" if provider == "anthropic" else "gpt-5"),
        )
        api_key = st.text_input(
            "API key",
            value="",
            type="password",
            help="Needed for the selected provider.",
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input("Max output tokens", min_value=200, max_value=16000, value=1600, step=100)
        num_runs = st.number_input("Number of runs", min_value=1, max_value=25, value=10, step=1)
        notes = st.text_input("Batch notes", value="")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Prompt and packet")
        prompt = st.text_area("Benchmark prompt", value=DEFAULT_PROMPT, height=220)

        packet_mode = st.radio(
            "Packet input mode",
            ["upload_file", "paste_text"],
            index=0,
            help="Use only plain-text packet input for controlled benchmark runs.",
        )

        packet_name: Optional[str] = None
        packet_text: str = ""

        if packet_mode == "upload_file":
            uploaded = st.file_uploader(
                "Upload benchmark packet as .txt or .md",
                type=["txt", "md"],
                accept_multiple_files=False,
            )
            if uploaded is not None:
                packet_name = uploaded.name
                packet_text = decode_packet_bytes(uploaded.read())
                st.info(f"Loaded packet: {packet_name}")
        else:
            packet_text = st.text_area("Paste packet text", value="", height=320)
            if packet_text.strip():
                packet_name = "pasted_packet.txt"
                packet_text = normalize_packet_text(packet_text)

        if packet_text.strip():
            packet_hash_preview = sha256_bytes(packet_text.encode("utf-8"))
            st.markdown("### Packet preview")
            st.code(packet_text[:1200], language="text")
            st.caption(f"Packet SHA256: {packet_hash_preview}")

            try:
                payload_preview = build_full_payload(prompt, packet_text, packet_name)
                st.markdown("### Full payload preview")
                st.code(payload_preview[:2000], language="text")
            except Exception as exc:
                st.error(str(exc))

        run_batch = st.button("Run benchmark batch", type="primary")

        if run_batch:
            if not api_key:
                st.error("Enter an API key.")
            elif not prompt.strip():
                st.error("Prompt cannot be empty.")
            elif not packet_text.strip():
                st.error("Packet text is required for this benchmark.")
            else:
                progress = st.progress(0)
                status = st.empty()
                successes = 0
                failures = []

                packet_hash = sha256_bytes(packet_text.encode("utf-8"))
                full_payload = build_full_payload(prompt, packet_text, packet_name)

                for i in range(int(num_runs)):
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i+1:02d}"
                    run_dir = outputs_dir / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    out_path = run_dir / f"{run_id}_output.txt"
                    meta_path = run_dir / f"{run_id}_meta.json"
                    packet_path = run_dir / f"{run_id}_packet.txt"
                    payload_path = run_dir / f"{run_id}_payload.txt"

                    try:
                        status.write(f"Running {i+1} of {int(num_runs)} ...")

                        save_text(packet_path, packet_text)
                        save_text(payload_path, full_payload)

                        if provider == "anthropic":
                            output_text = call_anthropic(
                                api_key=api_key,
                                model=model,
                                full_payload=full_payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )
                        elif provider == "openai":
                            output_text = call_openai(
                                api_key=api_key,
                                model=model,
                                full_payload=full_payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )
                        else:
                            raise RuntimeError("Unsupported provider for this benchmark.")

                        save_text(out_path, output_text)

                        meta = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "provider": provider,
                            "model": model,
                            "packet_name": packet_name,
                            "packet_sha256": packet_hash,
                            "packet_mode": packet_mode,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                            "notes": notes,
                            "packet_file": str(packet_path),
                            "payload_file": str(payload_path),
                            "output_file": str(out_path),
                        }
                        save_text(meta_path, json.dumps(meta, indent=2))

                        record = RunRecord(
                            run_id=run_id,
                            timestamp=meta["timestamp"],
                            provider=provider,
                            model=model,
                            prompt=prompt,
                            packet_name=packet_name,
                            packet_sha256=packet_hash,
                            packet_mode=packet_mode,
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            notes=notes,
                            output_file=str(out_path),
                            packet_file=str(packet_path),
                            payload_file=str(payload_path),
                        )
                        append_record(csv_path, record)
                        successes += 1

                    except Exception as exc:
                        failures.append(f"Run {i+1}: {exc}")

                    progress.progress((i + 1) / int(num_runs))
                    time.sleep(0.15)

                if successes:
                    st.success(f"Completed {successes} run(s).")
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Scoring log")
        df = load_records(csv_path)
        if df.empty:
            st.info("No runs logged yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("### Update Originality / manual notes")
            run_ids = df["run_id"].tolist()
            selected_run = st.selectbox("Select run", run_ids)
            current = df[df["run_id"] == selected_run].iloc[0]

            with st.form("score_form"):
                originality_label = st.text_input("Originality label", value=str(current.get("originality_label", "") or ""))
                originality_score = st.text_input(
                    "Originality score",
                    value="" if pd.isna(current.get("originality_score")) else str(current.get("originality_score")),
                )
                manual_rating = st.selectbox(
                    "Manual rating",
                    ["", "strong", "decent", "weak"],
                    index=["", "strong", "decent", "weak"].index(str(current.get("manual_rating", "") or ""))
                    if str(current.get("manual_rating", "") or "") in ["", "strong", "decent", "weak"]
                    else 0,
                )
                manual_notes = st.text_area("Manual notes", value=str(current.get("manual_notes", "") or ""), height=120)
                submitted = st.form_submit_button("Save score")
                if submitted:
                    parsed_score = None
                    raw = originality_score.strip()
                    if raw:
                        try:
                            parsed_score = float(raw)
                        except ValueError:
                            st.error("Originality score must be numeric if provided.")
                            st.stop()
                    update_record(
                        csv_path,
                        selected_run,
                        {
                            "originality_label": originality_label,
                            "originality_score": parsed_score,
                            "manual_rating": manual_rating,
                            "manual_notes": manual_notes,
                        },
                    )
                    st.success("Saved.")
                    st.rerun()

            output_path = Path(str(current["output_file"]))
            if output_path.exists():
                st.markdown("### Selected output")
                st.code(output_path.read_text(encoding="utf-8"), language="text")
                st.download_button(
                    "Download selected output",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime="text/plain",
                )

            packet_file = str(current.get("packet_file", "") or "")
            payload_file = str(current.get("payload_file", "") or "")

            if packet_file and Path(packet_file).exists():
                st.markdown("### Packet used for selected run")
                st.code(Path(packet_file).read_text(encoding="utf-8"), language="text")

            if payload_file and Path(payload_file).exists():
                st.markdown("### Payload sent for selected run")
                st.code(Path(payload_file).read_text(encoding="utf-8"), language="text")

            st.markdown("### Export batch")
            zip_bytes = export_zip(df, outputs_dir)
            st.download_button(
                "Download outputs + CSV",
                data=zip_bytes,
                file_name="benchmark_runs_export.zip",
                mime="application/zip",
            )

    st.markdown("---")
    st.subheader("Recommended use")
    st.write(
        "Use this to compare API-driven generations against Claude's web GUI. Keep the packet text, prompt, model, and generation settings fixed. "
        "Each run now saves the exact packet text and the full payload sent to the API, so the benchmark is auditable."
    )


if __name__ == "__main__":
    main()