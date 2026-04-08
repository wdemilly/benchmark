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

APP_TITLE = "Anthropic Benchmark Harness"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SYSTEM_PROMPT = (
    "You are writing fiction, not explaining it. Stay inside the local scene logic at all times. "
    "Do not summarize themes, do not generalize, do not add commentary before or after the scene, "
    "and do not step outside the narrator's lived moment. Keep the prose behavior governed by the "
    "packet's source passages and constraints. Favor concrete handling, local observation, and practical "
    "dialogue over interpretive phrasing. If a sentence begins to sound polished, explanatory, or neatly "
    "conclusive, prefer the plainer in-scene alternative."
)
DEFAULT_TASK_PROMPT = (
    "Read the benchmark packet straight through. Then write the benchmark scene only. "
    "First person, past tense. Keep to the packet's target length and constraints. Return plain text only."
)
ANCHOR_PROMPT = (
    "Read the packet carefully. In 5 short bullet points, state only the operational writing constraints "
    "you must obey while drafting. Do not summarize the story."
)

DATA_DIR = Path("benchmark_runs")
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "runs.csv"


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    model: str
    mode: str
    temperature: float
    max_tokens: int
    packet_name: str
    packet_sha256: str
    packet_file: str
    payload_file: str
    output_file: str
    notes: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_records() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "model",
            "mode",
            "temperature",
            "max_tokens",
            "packet_name",
            "packet_sha256",
            "packet_file",
            "payload_file",
            "output_file",
            "notes",
            "originality_label",
            "originality_score",
            "manual_rating",
            "manual_notes",
        ]
    )


def append_record(record: RunRecord) -> None:
    df = load_records()
    df = pd.concat([df, pd.DataFrame([asdict(record)])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


def update_record(run_id: str, updates: dict) -> None:
    df = load_records()
    if df.empty:
        return
    mask = df["run_id"] == run_id
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            df.loc[mask, key] = value
    df.to_csv(CSV_PATH, index=False)


def normalize_packet_text(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


def decode_packet_bytes(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return normalize_packet_text(text)


def packet_block(packet_name: str, packet_text: str) -> str:
    return f"<BENCHMARK_PACKET name=\"{packet_name}\">\n{packet_text}\n</BENCHMARK_PACKET>"


def anchor_constraints_text(text: str) -> str:
    parts: List[str] = []
    for block in getattr(text, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def build_single_turn_preview(system_prompt: str, packet_name: str, packet_text: str, task_prompt: str) -> str:
    return (
        "SYSTEM\n"
        f"{system_prompt}\n\n"
        "USER CONTENT BLOCK 1\n"
        f"{packet_block(packet_name, packet_text)}\n\n"
        "USER CONTENT BLOCK 2\n"
        f"<TASK>\n{task_prompt}\n</TASK>"
    )


def build_two_turn_preview(
    system_prompt: str,
    packet_name: str,
    packet_text: str,
    task_prompt: str,
    anchor_prompt: str,
    anchor_reply: str,
) -> str:
    return (
        "SYSTEM\n"
        f"{system_prompt}\n\n"
        "TURN 1 USER CONTENT BLOCK 1\n"
        f"{packet_block(packet_name, packet_text)}\n\n"
        "TURN 1 USER CONTENT BLOCK 2\n"
        f"{anchor_prompt}\n\n"
        "TURN 1 ASSISTANT\n"
        f"{anchor_reply}\n\n"
        "TURN 2 USER\n"
        f"{task_prompt}"
    )


def call_anthropic_single_turn(
    api_key: str,
    model: str,
    system_prompt: str,
    packet_name: str,
    packet_text: str,
    task_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": packet_block(packet_name, packet_text)},
                    {"type": "text", "text": f"<TASK>\n{task_prompt}\n</TASK>"},
                ],
            }
        ],
    )
    return anchor_constraints_text(resp)


def call_anthropic_two_turn(
    api_key: str,
    model: str,
    system_prompt: str,
    packet_name: str,
    packet_text: str,
    anchor_prompt: str,
    task_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)

    first = client.messages.create(
        model=model,
        system=system_prompt,
        temperature=0.7,
        max_tokens=250,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": packet_block(packet_name, packet_text)},
                    {"type": "text", "text": anchor_prompt},
                ],
            }
        ],
    )
    anchor_reply = anchor_constraints_text(first)

    second = client.messages.create(
        model=model,
        system=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": packet_block(packet_name, packet_text)},
                    {"type": "text", "text": anchor_prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": anchor_reply}]},
            {"role": "user", "content": [{"type": "text", "text": task_prompt}]},
        ],
    )
    return anchor_reply, anchor_constraints_text(second)


def export_zip(df: pd.DataFrame) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(OUTPUTS_DIR.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(DATA_DIR)))
    mem.seek(0)
    return mem.read()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Anthropic-only benchmark harness with auditable packet and payload saving.")

    with st.sidebar:
        st.header("Run setup")
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_key = st.text_input("Anthropic API key", value="", type="password")
        mode = st.selectbox("Request mode", ["single_turn", "two_turn_anchor"], index=1)
        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input("Max output tokens", min_value=200, max_value=16000, value=1600, step=100)
        num_runs = st.number_input("Number of runs", min_value=1, max_value=25, value=3, step=1)
        notes = st.text_input("Batch notes", value="")

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Prompt and packet")
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=180)
        task_prompt = st.text_area("Task prompt", value=DEFAULT_TASK_PROMPT, height=120)
        anchor_prompt = st.text_area("Anchor prompt", value=ANCHOR_PROMPT, height=100)

        packet_mode = st.radio("Packet input mode", ["upload_file", "paste_text"], index=0)
        packet_name = ""
        packet_text = ""

        if packet_mode == "upload_file":
            uploaded = st.file_uploader("Upload benchmark packet as .txt or .md", type=["txt", "md"])
            if uploaded is not None:
                packet_name = uploaded.name
                packet_text = decode_packet_bytes(uploaded.read())
                st.info(f"Loaded packet: {packet_name}")
        else:
            packet_name = "pasted_packet.txt"
            packet_text = normalize_packet_text(st.text_area("Paste packet text", value="", height=320))

        if packet_text.strip():
            st.markdown("### Packet preview")
            st.code(packet_text[:1500], language="text")
            st.caption(f"Packet SHA256: {sha256_bytes(packet_text.encode('utf-8'))}")

            st.markdown("### Request preview")
            preview = build_single_turn_preview(system_prompt, packet_name or "benchmark_packet.txt", packet_text, task_prompt)
            st.code(preview[:2500], language="text")

        run_batch = st.button("Run benchmark batch", type="primary")

        if run_batch:
            if not api_key:
                st.error("Enter an Anthropic API key.")
            elif not packet_text.strip():
                st.error("Packet text is required.")
            elif not system_prompt.strip() or not task_prompt.strip():
                st.error("System prompt and task prompt are required.")
            else:
                progress = st.progress(0)
                status = st.empty()
                successes = 0
                failures: list[str] = []
                packet_hash = sha256_bytes(packet_text.encode("utf-8"))
                packet_name = packet_name or "benchmark_packet.txt"

                for i in range(int(num_runs)):
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i+1:02d}"
                    run_dir = OUTPUTS_DIR / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    packet_file = run_dir / f"{run_id}_packet.txt"
                    payload_file = run_dir / f"{run_id}_payload.txt"
                    output_file = run_dir / f"{run_id}_output.txt"
                    meta_file = run_dir / f"{run_id}_meta.json"
                    anchor_file = run_dir / f"{run_id}_anchor.txt"

                    try:
                        status.write(f"Running {i+1} of {int(num_runs)} ...")
                        save_text(packet_file, packet_text)

                        if mode == "single_turn":
                            payload_text = build_single_turn_preview(system_prompt, packet_name, packet_text, task_prompt)
                            save_text(payload_file, payload_text)
                            output_text = call_anthropic_single_turn(
                                api_key=api_key,
                                model=model,
                                system_prompt=system_prompt,
                                packet_name=packet_name,
                                packet_text=packet_text,
                                task_prompt=task_prompt,
                                temperature=float(temperature),
                                max_tokens=int(max_tokens),
                            )
                        else:
                            anchor_reply, output_text = call_anthropic_two_turn(
                                api_key=api_key,
                                model=model,
                                system_prompt=system_prompt,
                                packet_name=packet_name,
                                packet_text=packet_text,
                                anchor_prompt=anchor_prompt,
                                task_prompt=task_prompt,
                                temperature=float(temperature),
                                max_tokens=int(max_tokens),
                            )
                            save_text(anchor_file, anchor_reply)
                            payload_text = build_two_turn_preview(
                                system_prompt,
                                packet_name,
                                packet_text,
                                task_prompt,
                                anchor_prompt,
                                anchor_reply,
                            )
                            save_text(payload_file, payload_text)

                        save_text(output_file, output_text)
                        meta = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "model": model,
                            "mode": mode,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                            "packet_name": packet_name,
                            "packet_sha256": packet_hash,
                            "packet_file": str(packet_file),
                            "payload_file": str(payload_file),
                            "output_file": str(output_file),
                            "anchor_file": str(anchor_file) if anchor_file.exists() else "",
                            "notes": notes,
                        }
                        save_text(meta_file, json.dumps(meta, indent=2))
                        append_record(
                            RunRecord(
                                run_id=run_id,
                                timestamp=meta["timestamp"],
                                model=model,
                                mode=mode,
                                temperature=float(temperature),
                                max_tokens=int(max_tokens),
                                packet_name=packet_name,
                                packet_sha256=packet_hash,
                                packet_file=str(packet_file),
                                payload_file=str(payload_file),
                                output_file=str(output_file),
                                notes=notes,
                            )
                        )
                        successes += 1
                    except Exception as exc:
                        failures.append(f"Run {i+1}: {exc}")

                    progress.progress((i + 1) / int(num_runs))
                    time.sleep(0.1)

                if successes:
                    st.success(f"Completed {successes} run(s).")
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Run log")
        df = load_records()
        if df.empty:
            st.info("No runs logged yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
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
                        parsed_score = float(raw)
                    update_record(
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
                st.markdown("### Output")
                st.code(output_path.read_text(encoding="utf-8"), language="text")

            packet_path = Path(str(current["packet_file"]))
            if packet_path.exists():
                st.markdown("### Packet used")
                st.code(packet_path.read_text(encoding="utf-8"), language="text")

            payload_path = Path(str(current["payload_file"]))
            if payload_path.exists():
                st.markdown("### Payload sent")
                st.code(payload_path.read_text(encoding="utf-8"), language="text")

            st.markdown("### Export batch")
            zip_bytes = export_zip(df)
            st.download_button(
                "Download outputs + CSV",
                data=zip_bytes,
                file_name="benchmark_runs_export.zip",
                mime="application/zip",
            )

    st.markdown("---")
    st.write(
        "This harness is designed to get closer to the Claude web environment by using a top-level system prompt, "
        "clear packet/task separation, and an optional two-turn anchor step. Each run saves the exact packet and "
        "payload for auditability."
    )


if __name__ == "__main__":
    main()
