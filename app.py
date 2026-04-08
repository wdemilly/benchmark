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


APP_TITLE = "Author Immersion Harness"
DATA_DIR = Path("author_immersion_runs")
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_AUTHOR_PROMPT = """You are not Claude. You are the author of the combined source texts document.

You wrote every passage in the combined source texts document. The character profiles are your notes. The outline is your plan for this chapter.

Read all attached documents from beginning to end. Do not sample them.

Then write the chapter from the outline exactly as you would write it yourself. Construct each sentence from within the habits of mind, sentence movement, and narrative logic already present in the source texts. Do not shift into explanatory prose, thematic summary, or polished generalization. Write the chapter straight through in one continuous pass, first sentence to last. Do not draft short and expand. Return plain text only, with no commentary.
"""


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    notes: str
    source_name: str = ""
    outline_name: str = ""
    profiles_name: str = ""
    source_sha256: str = ""
    outline_sha256: str = ""
    profiles_sha256: str = ""
    system_file: str = ""
    payload_file: str = ""
    output_file: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_text(text: str) -> str:
    return (
        text.replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


def decode_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode {uploaded_file.name}. Use a plain-text .txt or .md file."
            ) from exc
    return normalize_text(text)


def normalize_anthropic_text(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def load_records(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "provider",
            "model",
            "temperature",
            "max_tokens",
            "notes",
            "source_name",
            "outline_name",
            "profiles_name",
            "source_sha256",
            "outline_sha256",
            "profiles_sha256",
            "system_file",
            "payload_file",
            "output_file",
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


def build_payload(source_text: str, outline_text: str, profiles_text: str) -> str:
    clean_source = source_text.strip()
    clean_outline = outline_text.strip()
    clean_profiles = profiles_text.strip()

    if not clean_source:
        raise RuntimeError("Source text is required.")
    if not clean_outline:
        raise RuntimeError("Outline text is required.")

    payload_parts = [
        "BEGIN COMBINED SOURCE TEXTS",
        clean_source,
        "END COMBINED SOURCE TEXTS",
        "",
        "BEGIN OUTLINE",
        clean_outline,
        "END OUTLINE",
    ]

    if clean_profiles:
        payload_parts.extend([
            "",
            "BEGIN CHARACTER PROFILES",
            clean_profiles,
            "END CHARACTER PROFILES",
        ])

    return "\n".join(payload_parts)


def call_anthropic(
    api_key: str,
    model: str,
    system_prompt: str,
    payload: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    client = anthropic.Anthropic(api_key=api_key)

    resp = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": payload,
            }
        ],
    )
    return normalize_anthropic_text(resp)


def call_openai(
    api_key: str,
    model: str,
    system_prompt: str,
    payload: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=payload,
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


def read_text_if_exists(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload source text, outline, and optional profiles, then run author-immersion generations.")

    csv_path = DATA_DIR / "runs.csv"
    outputs_dir = DATA_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    with st.sidebar:
        st.header("Run setup")
        provider = st.selectbox("Provider", ["anthropic", "openai"], index=0)
        default_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5"
        model = st.text_input("Model", value=default_model)
        api_key = st.text_input("API key", value="", type="password")
        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input("Max output tokens", min_value=200, max_value=20000, value=3000, step=100)
        num_runs = st.number_input("Number of runs", min_value=1, max_value=25, value=3, step=1)
        notes = st.text_input("Batch notes", value="")
        st.markdown("---")
        st.caption("Use plain-text .txt or .md files for reliable upload behavior.")

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Prompt and materials")

        system_prompt = st.text_area(
            "Author-immersion prompt",
            value=DEFAULT_AUTHOR_PROMPT,
            height=260,
        )

        source_text = ""
        outline_text = ""
        profiles_text = ""

        source_name = ""
        outline_name = ""
        profiles_name = ""

        st.markdown("### Combined source texts")
        source_mode = st.radio("Source input mode", ["upload_file", "paste_text"], index=0, key="source_mode")
        if source_mode == "upload_file":
            uploaded_source = st.file_uploader(
                "Upload combined source texts",
                type=["txt", "md"],
                accept_multiple_files=False,
                key="source_upload",
            )
            if uploaded_source is not None:
                source_name = uploaded_source.name
                source_text = decode_uploaded_text(uploaded_source)
                st.info(f"Loaded source text: {source_name}")
        else:
            source_name = "pasted_source.txt"
            source_text = normalize_text(
                st.text_area("Paste combined source texts", value="", height=220, key="source_paste")
            )

        st.markdown("### Outline")
        outline_mode = st.radio("Outline input mode", ["upload_file", "paste_text"], index=0, key="outline_mode")
        if outline_mode == "upload_file":
            uploaded_outline = st.file_uploader(
                "Upload outline",
                type=["txt", "md"],
                accept_multiple_files=False,
                key="outline_upload",
            )
            if uploaded_outline is not None:
                outline_name = uploaded_outline.name
                outline_text = decode_uploaded_text(uploaded_outline)
                st.info(f"Loaded outline: {outline_name}")
        else:
            outline_name = "pasted_outline.txt"
            outline_text = normalize_text(
                st.text_area("Paste outline", value="", height=180, key="outline_paste")
            )

        st.markdown("### Character profiles (optional)")
        profiles_mode = st.radio(
            "Profiles input mode",
            ["none", "upload_file", "paste_text"],
            index=0,
            key="profiles_mode",
        )
        if profiles_mode == "upload_file":
            uploaded_profiles = st.file_uploader(
                "Upload character profiles",
                type=["txt", "md"],
                accept_multiple_files=False,
                key="profiles_upload",
            )
            if uploaded_profiles is not None:
                profiles_name = uploaded_profiles.name
                profiles_text = decode_uploaded_text(uploaded_profiles)
                st.info(f"Loaded character profiles: {profiles_name}")
        elif profiles_mode == "paste_text":
            profiles_name = "pasted_profiles.txt"
            profiles_text = normalize_text(
                st.text_area("Paste character profiles", value="", height=160, key="profiles_paste")
            )

        if source_text.strip():
            st.markdown("### Source preview")
            st.code(source_text[:1500], language="text")
            st.caption(f"Source SHA256: {sha256_bytes(source_text.encode('utf-8'))}")

        if outline_text.strip():
            st.markdown("### Outline preview")
            st.code(outline_text[:1200], language="text")
            st.caption(f"Outline SHA256: {sha256_bytes(outline_text.encode('utf-8'))}")

        if profiles_text.strip():
            st.markdown("### Profiles preview")
            st.code(profiles_text[:1000], language="text")
            st.caption(f"Profiles SHA256: {sha256_bytes(profiles_text.encode('utf-8'))}")

        if source_text.strip() and outline_text.strip():
            try:
                payload_preview = build_payload(source_text, outline_text, profiles_text)
                st.markdown("### Payload preview")
                st.code(payload_preview[:2500], language="text")
            except Exception as exc:
                st.error(str(exc))

        run_batch = st.button("Run batch", type="primary")

        if run_batch:
            if not api_key:
                st.error("Enter an API key.")
            elif not system_prompt.strip():
                st.error("Prompt cannot be empty.")
            elif not source_text.strip():
                st.error("Combined source texts are required.")
            elif not outline_text.strip():
                st.error("Outline is required.")
            else:
                source_hash = sha256_bytes(source_text.encode("utf-8"))
                outline_hash = sha256_bytes(outline_text.encode("utf-8"))
                profiles_hash = sha256_bytes(profiles_text.encode("utf-8")) if profiles_text.strip() else ""
                payload = build_payload(source_text, outline_text, profiles_text)

                progress = st.progress(0)
                status = st.empty()
                successes = 0
                failures = []

                for i in range(int(num_runs)):
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i+1:02d}"
                    run_dir = outputs_dir / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    source_path = run_dir / f"{run_id}_source.txt"
                    outline_path = run_dir / f"{run_id}_outline.txt"
                    profiles_path = run_dir / f"{run_id}_profiles.txt"
                    system_path = run_dir / f"{run_id}_system_prompt.txt"
                    payload_path = run_dir / f"{run_id}_payload.txt"
                    output_path = run_dir / f"{run_id}_output.txt"
                    meta_path = run_dir / f"{run_id}_meta.json"

                    try:
                        status.write(f"Running {i+1} of {int(num_runs)}...")

                        save_text(source_path, source_text)
                        save_text(outline_path, outline_text)
                        if profiles_text.strip():
                            save_text(profiles_path, profiles_text)
                        save_text(system_path, system_prompt)
                        save_text(payload_path, payload)

                        if provider == "anthropic":
                            output_text = call_anthropic(
                                api_key=api_key,
                                model=model,
                                system_prompt=system_prompt,
                                payload=payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )
                        else:
                            output_text = call_openai(
                                api_key=api_key,
                                model=model,
                                system_prompt=system_prompt,
                                payload=payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )

                        save_text(output_path, output_text)

                        meta = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "provider": provider,
                            "model": model,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                            "notes": notes,
                            "source_name": source_name,
                            "outline_name": outline_name,
                            "profiles_name": profiles_name,
                            "source_sha256": source_hash,
                            "outline_sha256": outline_hash,
                            "profiles_sha256": profiles_hash,
                            "source_file": str(source_path),
                            "outline_file": str(outline_path),
                            "profiles_file": str(profiles_path) if profiles_text.strip() else "",
                            "system_file": str(system_path),
                            "payload_file": str(payload_path),
                            "output_file": str(output_path),
                        }
                        save_text(meta_path, json.dumps(meta, indent=2))

                        record = RunRecord(
                            run_id=run_id,
                            timestamp=meta["timestamp"],
                            provider=provider,
                            model=model,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                            notes=notes,
                            source_name=source_name,
                            outline_name=outline_name,
                            profiles_name=profiles_name,
                            source_sha256=source_hash,
                            outline_sha256=outline_hash,
                            profiles_sha256=profiles_hash,
                            system_file=str(system_path),
                            payload_file=str(payload_path),
                            output_file=str(output_path),
                        )
                        append_record(csv_path, record)
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
        df = load_records(csv_path)

        if df.empty:
            st.info("No runs logged yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("### Update scoring")
            run_ids = df["run_id"].tolist()
            selected_run = st.selectbox("Select run", run_ids)
            current = df[df["run_id"] == selected_run].iloc[0]

            with st.form("score_form"):
                originality_label = st.text_input(
                    "Originality label",
                    value=str(current.get("originality_label", "") or ""),
                )
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
                manual_notes = st.text_area(
                    "Manual notes",
                    value=str(current.get("manual_notes", "") or ""),
                    height=120,
                )
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

            st.markdown("### Selected run files")

            output_text = read_text_if_exists(str(current.get("output_file", "") or ""))
            if output_text:
                st.markdown("#### Output")
                st.code(output_text, language="text")

            system_text = read_text_if_exists(str(current.get("system_file", "") or ""))
            if system_text:
                st.markdown("#### System prompt")
                st.code(system_text, language="text")

            payload_text = read_text_if_exists(str(current.get("payload_file", "") or ""))
            if payload_text:
                st.markdown("#### Payload")
                st.code(payload_text[:4000], language="text")

            st.markdown("### Export batch")
            zip_bytes = export_zip(df, outputs_dir)
            st.download_button(
                "Download outputs + CSV",
                data=zip_bytes,
                file_name="author_immersion_runs_export.zip",
                mime="application/zip",
            )

    st.markdown("---")
    st.subheader("Recommended use")
    st.write(
        "Upload the combined source texts and the actual outline. Character profiles are optional. "
        "This app saves the exact source text, outline, profiles, system prompt, payload, and output for each run."
    )


if __name__ == "__main__":
    main()