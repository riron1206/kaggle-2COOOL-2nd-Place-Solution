import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


INPUT_IMAGES: List[str] = [
    "gdrive/images/0.jpg",
]
OUTPUT_JSON: str = "vllm_multi_image_result.json"
DEFAULT_API_BASE: str = "http://localhost:8000/v1"
DEFAULT_MODEL: str = "/data/models/zai-org/GLM-4.5V"

RETRY_ON_EMPTY_MAX_ATTEMPTS: int = 5
RETRY_ON_EMPTY_STEP: float = 0.1
RETRY_ON_EMPTY_CAP: float = 1.5

PROMPT_COUNT: str = """
You are an assistant that analyzes road anomalies and traffic incidents from multiple images.
The input consists of consecutive frames extracted from a single dashcam video (earliest → latest). Use temporal evolution across frames.

Work in **ULTRATHINK** mode.

Output rules (STRICT):
- Return pure JSON only — no code fences or extra text.
- Output EXACTLY these 9 keys (no more, no less).
- Use integers (not strings) for numeric fields. Use one of the allowed strings for textual fields.
- Count ONLY participants directly involved in the anomaly/incident (exclude bystanders).
- If nothing abnormal is visible, use: Incident Detection = -1, Crash Severity = "0. No Crash", Ego-car involved = 0, Label = "unknown", all counts = 0.

JSON format:
{
  "Incident window start frame": <int: -1..N-1>,
  "Incident Detection": <int: -1|0|1>,
  "Crash Severity": "<one allowed string>",
  "Ego-car involved": <int: 0|1>,
  "Label": "<one allowed string>",
  "Number of Bicyclists/Scooters": <int >= 0>,
  "Number of animals involved": <int >= 0>,
  "Number of pedestrians involved": <int >= 0>,
  "Number of vehicles involved (excluding ego-car)": <int >= 0>
}

Allowed values:

Incident window start frame (Numeric):
- It is the index of the input image where the hazard/accident first becomes visibly evident.
- If "Incident Detection" ∈ {0,1}, pick an integer in [0, N-1].
- If "Incident Detection" = -1 (no anomaly visible), set it to -1.

Incident Detection (Numeric):
  1 = Accident
  0 = Hazard (no collision; e.g., debris, animal on road, flooded road, near-miss)
  -1 = No incident

Crash Severity (Textual; choose EXACTLY one):
  "0. No Crash"
  "1. Ego-car collided but did not stop"
  "2. Ego-car collided and could not continue moving"
  "3. Ego-car collided with at-least one person or cyclist"
  "4. Other cars collided with person/car/object but ego-car is ok"
  "5. Multiple vehicles collided with ego-car"
  "6. One or Multiple vehicles collided but ego-car is fine"

Ego-car involved (Numeric):
  0 = Not involved (ego-car does not collide or directly interact with the hazard)
  1 = Involved (ego-car collides, is hit, or physically interacts with the hazard)

Label (Textual; choose EXACTLY one):
  "ego-car hits barrier"
  "flying object hit the car"
  "ego-car hit an animal"
  "many cars/pedestrians/cyclists collided"
  "car hits barrier"
  "ego-car hits a pedestrian"
  "animal on the road"
  "car flipped over"
  "ego-car hits a crossing cyclist"
  "vehicle drives into another vehicle"
  "ego-car loses control"
  "scooter on the road"
  "bicycle on road"
  "pedestrian is crossing the street"
  "pedestrian on the road"
  "vehicle hits ego-car"
  "ego-car hits a vehicle"
  "vehicle overtakes"
  "unknown"

Counting rules:
- Number of Bicyclists/Scooters: cyclists or scooters directly involved.
- Number of animals involved: animals directly involved (e.g., collision or blocking the lane).
- Number of pedestrians involved: pedestrians directly involved (e.g., in path/impact/near-miss).
- Number of vehicles involved (excluding ego-car): vehicles (cars/trucks/buses/vans/motorcycles) involved, excluding the ego-car.

Decision guidelines:
- Set Incident Detection = 1 (Accident) if any visible collision occurs.
- Use 0 (Hazard) for non-contact anomalies (debris, flooded road, animal on road, downed power line, fog/smoke, fire) or clear near-miss.
- Map Crash Severity consistently with the observed involvement. If no collision, use "0. No Crash".
- Choose the most specific Label string that matches the visible evidence; if unclear, use "unknown".
""".strip()

PROMPT_TEXT: str = """
You are an assistant that analyzes road anomalies and traffic incidents from multiple images.
The input consists of consecutive frames extracted from a single dashcam video (earliest → latest). Use temporal evolution across frames.

Work in **ULTRATHINK** mode.

Style targets (to match the desired granularity and maximize CIDEr-D / METEOR / SPICE):
- ONE sentence per key, 10–18 words, present tense, active voice, concise and concrete.
- Prefer normalized vocabulary: ego-car, car, vehicle, truck, bus, van, motorcycle, bicycle, pedestrian, crosswalk, intersection, lane, oncoming traffic, traffic light (red/green/yellow), debris, flooded road, snow, ice, fog, smoke, fire.
- Mention salient objects, clear counts if visible, spatial relations (left/right/ahead/oncoming/at intersection), and actions (turns, brakes, stops, merges, collides, cuts in).
- Avoid speculation and rare synonyms (no “likely”, “appears”, “seems”).

Output rules:
- Use exactly the two keys below and return pure JSON only.

JSON format:
{
  "Incident window start frame": <int: -1..N-1>,
  "Caption Before Incident": "<objective description of the last stable scene just before the anomaly or near-miss>",
  "Reason of Incident": "<cause-and-effect naming concrete actors or hazards; if novel/unclear, write 'unknown hazard: <visible evidence>'>"
}

Definitions:
- "Incident window start frame": The index of the input image where the hazard/accident first becomes visibly evident. If no anomaly visible, set it to -1.
- "Caption Before Incident": Concise, objective scene description immediately before the anomaly (e.g., “Ego-car drives on a city road with heavy traffic.”).
- "Reason of Incident": Concise cause-and-effect explanation using normalized terms (e.g., “A vehicle turns left across oncoming traffic and collides with another vehicle.”).
""".strip()

PROMPT_RECONCILE: str = """
You are an assistant that reconciles two prior predictions using the provided images as ground truth.

Work in **ULTRATHINK** mode.

Inputs:
- Images: consecutive frames from a single dashcam video (earliest → latest). Use temporal evidence across frames.
- First prediction: Counts JSON.
- Second prediction: Texts JSON.

Goal:
- Produce ONE self-consistent JSON with EXACTLY the 11 keys below, corrected by image evidence whenever the two predictions contradict.

Return format (STRICT):
- Return pure JSON only — no code fences or extra text.
- Use EXACTLY these 11 keys (spelling and order do not matter, but keys must match exactly):
  1) "Incident window start frame"
  2) "Incident Detection"
  3) "Crash Severity"
  4) "Ego-car involved"
  5) "Label"
  6) "Number of Bicyclists/Scooters"
  7) "Number of animals involved"
  8) "Number of pedestrians involved"
  9) "Number of vehicles involved (excluding ego-car)"
  10) "Caption Before Incident"
  11) "Reason of Incident"
- Numeric fields must be integers. Text fields must be chosen from the allowed sets below (except the two caption fields).
- Base everything on visible evidence in the images. Do not speculate.

Allowed values (reuse these EXACT strings):

Incident window start frame (Numeric):
  The index of the input image where the hazard/accident first becomes visibly evident. If no anomaly visible, set it to -1.

Incident Detection (Numeric):
  1 = Accident
  0 = Hazard (no collision; e.g., debris/animal on road, flooded road, near-miss)
  -1 = No incident

Crash Severity (Textual; choose EXACTLY one):
  "0. No Crash"
  "1. Ego-car collided but did not stop"
  "2. Ego-car collided and could not continue moving"
  "3. Ego-car collided with at-least one person or cyclist"
  "4. Other cars collided with person/car/object but ego-car is ok"
  "5. Multiple vehicles collided with ego-car"
  "6. One or Multiple vehicles collided but ego-car is fine"

Label (Textual; choose EXACTLY one):
  "ego-car hits barrier"
  "flying object hit the car"
  "ego-car hit an animal"
  "many cars/pedestrians/cyclists collided"
  "car hits barrier"
  "ego-car hits a pedestrian"
  "animal on the road"
  "car flipped over"
  "ego-car hits a crossing cyclist"
  "vehicle drives into another vehicle"
  "ego-car loses control"
  "scooter on the road"
  "bicycle on road"
  "pedestrian is crossing the street"
  "pedestrian on the road"
  "vehicle hits ego-car"
  "ego-car hits a vehicle"
  "vehicle overtakes"
  "unknown"

Style for the two text fields:
- "Caption Before Incident": ONE concise sentence (10–18 words), present tense, objective description of the last stable scene before the anomaly/near-miss.
- "Reason of Incident": ONE concise sentence (10–18 words) explaining visible cause-and-effect using normalized vocabulary (car, vehicle, pedestrian, cyclist, crosswalk, lane, oncoming traffic, red/green light, debris, wet road, night). If there is no change from Texts JSON, use the value of the Texts JSON as the reasonof the incident.

Evidence precedence (when predictions contradict):
Images > Texts JSON > Counts JSON.

Global consistency rules (MUST satisfy all that apply):
A) If "Incident Detection" = -1:
   - "Crash Severity" = "0. No Crash"
   - "Ego-car involved" = 0
   - "Label" = "unknown"
   - All counts = 0
B) If "Incident Detection" = 0 (Hazard, no collision):
   - "Crash Severity" = "0. No Crash"
   - Default "Ego-car involved" = 0 (unless images show direct physical interaction with the hazard).
   - Choose a hazard-compatible "Label":
     {"animal on the road","bicycle on road","scooter on the road",
      "pedestrian is crossing the street","pedestrian on the road",
      "vehicle overtakes","unknown"}.
C) If "Incident Detection" = 1 (Accident):
   - Set "Ego-car involved" based on images.
   - If "Ego-car involved" = 1:
       * If collision with a person/cyclist → "Crash Severity" = "3. Ego-car collided with at-least one person or cyclist".
       * Else if ego continues to move → "Crash Severity" = "1. Ego-car collided but did not stop".
       * Else if ego cannot continue → "Crash Severity" = "2. Ego-car collided and could not continue moving".
       * "Label" must be one of:
         {"ego-car hits barrier","ego-car hits a pedestrian","ego-car hits a crossing cyclist",
          "ego-car hits a vehicle","vehicle hits ego-car","ego-car hit an animal",
          "flying object hit the car","ego-car loses control"}.
   - If "Ego-car involved" = 0:
       * If other vehicles collide with a person/cyclist/object → "Crash Severity" = "4. Other cars collided with person/car/object but ego-car is ok".
       * If collision is vehicle↔vehicle only (no person) → "Crash Severity" = "6. One or Multiple vehicles collided but ego-car is fine".
       * "Label" should match images; common choices:
         {"vehicle drives into another vehicle","car hits barrier","car flipped over",
          "many cars/pedestrians/cyclists collided","unknown"}.

Counting alignment rules (images first, then texts):
- Count ONLY participants directly involved in the anomaly/incident (exclude bystanders).
- If "Reason of Incident" mentions "pedestrian/cyclist/animal", set corresponding count ≥ 1 (if visible).
- If "Label" = "vehicle drives into another vehicle" and ego is not involved, set "Number of vehicles involved (excluding ego-car)" ≥ 2.
- If "Label" mentions ego-car interactions, ensure "Ego-car involved" = 1.

Sanity checks (fix before finalizing):
- "Incident Detection" = 1 ⇒ "Crash Severity" ≠ "0. No Crash".
- "Incident Detection" ∈ {0, -1} ⇒ "Crash Severity" = "0. No Crash".
- If any count > 0, "Incident Detection" ∈ {0,1}.
- Avoid contradictions between captions/reason and counts/labels.

Output:
- Return ONLY the final reconciled JSON with the 11 keys.
""".strip()

JSN_KEYS: List[str] = [
    "Incident window start frame",
    "Incident Detection",
    "Crash Severity",
    "Ego-car involved",
    "Label",
    "Number of Bicyclists/Scooters",
    "Number of animals involved",
    "Number of pedestrians involved",
    "Number of vehicles involved (excluding ego-car)",
    "Caption Before Incident",
    "Reason of Incident",
]

COUNT_KEYS: List[str] = [
    "Incident window start frame",
    "Incident Detection",
    "Crash Severity",
    "Ego-car involved",
    "Label",
    "Number of Bicyclists/Scooters",
    "Number of animals involved",
    "Number of pedestrians involved",
    "Number of vehicles involved (excluding ego-car)",
]

TEXT_KEYS: List[str] = [
    "Incident window start frame",
    "Caption Before Incident",
    "Reason of Incident",
]


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _find_balanced_json_block(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_json_robust(text: str) -> Dict[str, Any]:
    s = _strip_code_fence(text)
    try:
        return json.loads(s)
    except Exception:
        pass

    block = _find_balanced_json_block(s)
    if block:
        try:
            return json.loads(block)
        except Exception:
            pass

    s2 = s.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(s2)
    except Exception:
        return {"raw_output": text}


def post_validate(
    d: Dict[str, Any], expected_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    keys = expected_keys or JSN_KEYS
    allowed_numeric = {
        "Incident window start frame",
        "Incident Detection",
        "Ego-car involved",
        "Number of Bicyclists/Scooters",
        "Number of animals involved",
        "Number of pedestrians involved",
        "Number of vehicles involved (excluding ego-car)",
    }
    numeric_keys = set(keys).intersection(allowed_numeric)
    count_keys = set(keys).intersection(
        {
            "Number of Bicyclists/Scooters",
            "Number of animals involved",
            "Number of pedestrians involved",
            "Number of vehicles involved (excluding ego-car)",
        }
    )

    def _coerce_int(x: Any) -> int:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float):
            try:
                return int(round(x))
            except Exception:
                return 0
        if isinstance(x, str):
            xs = x.strip()
            m = re.match(r"^-?\d+", xs)
            if m:
                try:
                    return int(m.group(0))
                except Exception:
                    return 0
        return 0

    out: Dict[str, Any] = {}
    for k in keys:
        v = d.get(k, None)
        if k in numeric_keys:
            out[k] = _coerce_int(v)
        else:
            if v is None:
                out[k] = ""
            else:
                s = v if isinstance(v, str) else str(v)
                out[k] = s.strip()

    if "Incident Detection" in out and out["Incident Detection"] not in (-1, 0, 1):
        out["Incident Detection"] = 0
    if "Ego-car involved" in out and out["Ego-car involved"] not in (0, 1):
        out["Ego-car involved"] = 0
    for ck in count_keys:
        if out.get(ck, 0) < 0:
            out[ck] = 0

    return out


def detect_contradictions(
    counts_json: Dict[str, Any], texts_json: Dict[str, Any]
) -> List[str]:
    """
    counts_json（数値系8キー）と texts_json（テキスト2キー）の内容に矛盾がないかを簡易ルールでチェックし、矛盾メッセージのリストを返します（空リストなら矛盾なし）。
    """
    issues: List[str] = []

    try:
        incident = counts_json.get("Incident Detection", None)
        if isinstance(incident, str):
            try:
                incident = int(incident)
            except Exception:
                incident = None
    except Exception:
        incident = None

    crash = str(counts_json.get("Crash Severity", ""))
    try:
        ego_involved = counts_json.get("Ego-car involved", None)
        if isinstance(ego_involved, str):
            try:
                ego_involved = int(ego_involved)
            except Exception:
                ego_involved = None
    except Exception:
        ego_involved = None

    caption = str(texts_json.get("Caption Before Incident", ""))
    reason = str(texts_json.get("Reason of Incident", ""))
    text_all = (caption + "\n" + reason).lower()

    hazard_kw = [
        "collision",
        "collide",
        "crash",
        "hit",
        "near-miss",
        "hazard",
        "debris",
        "animal",
        "blocked",
        "overtake",
        "cut in",
        "skid",
        "lose control",
        "fire",
        "smoke",
        "fog",
        "flood",
        "pedestrian",
        "cyclist",
    ]

    if incident == 1 and crash.strip() == "0. No Crash":
        issues.append("Incident=1 but Crash Severity is '0. No Crash'")
    if ego_involved == 1 and crash.strip() == "0. No Crash":
        issues.append("Ego involved=1 but Crash Severity is '0. No Crash'")
    if incident == -1:
        if any(k in text_all for k in hazard_kw):
            issues.append("Texts mention hazard/collision while Incident Detection=-1")

    return issues


def to_file_url(path: str) -> str:
    p = Path(path).resolve()
    return p.as_uri()


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return base64.b64encode(b).decode("ascii")


def guess_image_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"


def has_empty_values(d: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        v = d.get(k, None)
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
    return False


def chat_with_images(
    api_base: str,
    api_key: str,
    model: str,
    image_urls: List[str],
    prompt: str,
    json_mode: bool,
    request_timeout: float,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
) -> Tuple[str, str]:
    client = OpenAI(base_url=api_base, api_key=api_key)
    image_parts = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_parts,
            ],
        }
    ]

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    kwargs["extra_body"] = {
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    }

    resp = client.chat.completions.create(**kwargs)
    print("[Debug]:", resp.usage.completion_tokens)

    content: str = ""
    reasoning_content: str = ""
    try:
        content = resp.choices[0].message.content or ""
    except Exception:
        content = ""

    try:
        rc = getattr(resp.choices[0].message, "reasoning_content", None)
        if rc:
            reasoning_content = rc
    except Exception:
        pass
    try:
        rc2 = getattr(resp.choices[0], "reasoning_content", None)
        if rc2 and not reasoning_content:
            reasoning_content = rc2
    except Exception:
        pass

    print("===== reasoning_content (from vLLM) =====")
    print(reasoning_content)
    print("===== content (from vLLM) =====")
    print(content)

    return content, reasoning_content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        nargs="+",
        default=INPUT_IMAGES,
        help="入力画像へのパスを複数指定（.jpg/.png 等）",
    )
    ap.add_argument(
        "--reconcile",
        action="store_true",
        help="PROMPT_COUNT と PROMPT_TEXT の2段推論後、矛盾時に再統合推論する",
    )
    ap.add_argument(
        "--tmp_json",
        default=None,
        help="中間JSONを保存するパス（counts/texts/reconciled と各content/reasoning_contentを保存）",
    )
    ap.add_argument("--output_json", default=OUTPUT_JSON, help="保存先 JSON")
    ap.add_argument(
        "--api_base",
        default=DEFAULT_API_BASE,
        help="vLLM OpenAI 互換 API のベース URL（例: http://localhost:8000/v1）",
    )
    ap.add_argument(
        "--api_key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API キー（未使用環境では 'EMPTY'）",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="served-model-name と合わせる（例: glm-4.5v）",
    )
    ap.add_argument(
        "--send_mode",
        choices=["file_url", "base64"],
        default="file_url",
        help="画像の渡し方: file_url= file:///... を渡す / base64= data:image/...;base64,... を渡す",
    )
    ap.add_argument("--max_tokens", type=int, default=8192, help="生成トークン上限")
    ap.add_argument("--temperature", type=float, default=0.2, help="サンプリング温度")
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="繰り返しペナルティ（>1 で抑制）",
    )
    ap.add_argument("--top_p", type=float, default=0.95, help="核サンプリング上限確率")
    ap.add_argument("--top_k", type=int, default=0, help="上位 k 語からサンプリング")
    ap.add_argument(
        "--timeout", type=float, default=600.0, help="HTTP リクエストのタイムアウト秒"
    )
    ap.add_argument(
        "--json_mode",
        action="store_true",
        help="response_format={'type':'json_object'} を有効化",
    )
    ap.add_argument(
        "--counts_only",
        action="store_true",
        help="Counts 推論のみ実行し、Counts キーだけを出力して終了",
    )
    args = ap.parse_args()

    image_paths = [str(Path(p).resolve()) for p in args.images]
    image_urls: List[str] = []
    if args.send_mode == "file_url":
        image_urls = [to_file_url(p) for p in image_paths]
    else:
        for p in image_paths:
            b64 = encode_image_base64(p)
            mime = guess_image_mime(p)
            image_urls.append(f"data:{mime};base64,{b64}")

    try:
        video_id = int(Path(image_paths[0]).parent.stem)
    except Exception:
        m = re.search(r"\d+", Path(image_paths[0]).parent.stem)
        video_id = int(m.group(0)) if m else 0

    counts_text, counts_reason = chat_with_images(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        image_urls=image_urls,
        prompt=PROMPT_COUNT,
        json_mode=args.json_mode,
        request_timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    counts_json_raw = parse_json_robust(counts_text)
    counts_json = post_validate(counts_json_raw, expected_keys=COUNT_KEYS)

    if has_empty_values(counts_json, COUNT_KEYS):
        base_rp = args.repetition_penalty
        for i in range(RETRY_ON_EMPTY_MAX_ATTEMPTS):
            new_rp = min(base_rp + (i + 1) * RETRY_ON_EMPTY_STEP, RETRY_ON_EMPTY_CAP)
            print(f"[Retry] Counts has empty fields. repetition_penalty={new_rp}")
            counts_text, counts_reason = chat_with_images(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                image_urls=image_urls,
                prompt=PROMPT_COUNT,
                json_mode=args.json_mode,
                request_timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                repetition_penalty=new_rp,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            counts_json_raw = parse_json_robust(counts_text)
            counts_json = post_validate(counts_json_raw, expected_keys=COUNT_KEYS)
            if not has_empty_values(counts_json, COUNT_KEYS):
                break

    if args.counts_only:
        if args.tmp_json:
            tmp_path = Path(args.tmp_json)
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = {
                "counts": counts_json,
                "counts_raw": {
                    "content": counts_text,
                    "reasoning_content": counts_reason,
                },
            }
            tmp_path.write_text(
                json.dumps(tmp, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        counts_out = counts_json
        out_path = Path(args.output_json)
        out_path.write_text(
            json.dumps(counts_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("===== final_json =====")
        print(counts_out)
        print(f"Saved JSON to {out_path.resolve()}")
        return

    texts_text, texts_reason = chat_with_images(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        image_urls=image_urls,
        prompt=PROMPT_TEXT,
        json_mode=args.json_mode,
        request_timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    texts_json_raw = parse_json_robust(texts_text)
    texts_json = post_validate(texts_json_raw, expected_keys=TEXT_KEYS)

    if has_empty_values(texts_json, TEXT_KEYS):
        base_rp = args.repetition_penalty
        for i in range(RETRY_ON_EMPTY_MAX_ATTEMPTS):
            new_rp = min(base_rp + (i + 1) * RETRY_ON_EMPTY_STEP, RETRY_ON_EMPTY_CAP)
            print(f"[Retry] Texts has empty fields. repetition_penalty={new_rp}")
            texts_text, texts_reason = chat_with_images(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                image_urls=image_urls,
                prompt=PROMPT_TEXT,
                json_mode=args.json_mode,
                request_timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                repetition_penalty=new_rp,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            texts_json_raw = parse_json_robust(texts_text)
            texts_json = post_validate(texts_json_raw, expected_keys=TEXT_KEYS)
            if not has_empty_values(texts_json, TEXT_KEYS):
                break

    if args.tmp_json:
        tmp_path = Path(args.tmp_json)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = {
            "counts": counts_json,
            "counts_raw": {
                "content": counts_text,
                "reasoning_content": counts_reason,
            },
            "texts": texts_json,
            "texts_raw": {
                "content": texts_text,
                "reasoning_content": texts_reason,
            },
        }
        tmp_path.write_text(
            json.dumps(tmp, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    final_json: Dict[str, Any] = {}

    if True:
        reconcile_context = json.dumps(
            {
                "First prediction (Counts JSON)": counts_json,
                "Second prediction (Texts JSON)": texts_json,
            },
            ensure_ascii=False,
        )

        reconcile_prompt = PROMPT_RECONCILE + "\n\n" + reconcile_context
        reconciled_text, reconciled_reason = chat_with_images(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            image_urls=image_urls,
            prompt=reconcile_prompt,
            json_mode=args.json_mode,
            request_timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        reconciled_json_raw = parse_json_robust(reconciled_text)
        final_json = post_validate(reconciled_json_raw)

        if has_empty_values(final_json, JSN_KEYS):
            base_rp = args.repetition_penalty
            for i in range(RETRY_ON_EMPTY_MAX_ATTEMPTS):
                new_rp = min(
                    base_rp + (i + 1) * RETRY_ON_EMPTY_STEP, RETRY_ON_EMPTY_CAP
                )
                print(
                    f"[Retry] Reconciled has empty fields. repetition_penalty={new_rp}"
                )
                reconciled_text, reconciled_reason = chat_with_images(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    image_urls=image_urls,
                    prompt=reconcile_prompt,
                    json_mode=args.json_mode,
                    request_timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    repetition_penalty=new_rp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                reconciled_json_raw = parse_json_robust(reconciled_text)
                final_json = post_validate(reconciled_json_raw)
                if not has_empty_values(final_json, JSN_KEYS):
                    break

        if args.tmp_json:
            tmp_path = Path(args.tmp_json)
            tmp = json.loads(tmp_path.read_text(encoding="utf-8"))
            tmp["reconciled"] = final_json
            tmp["reconciled_raw"] = {
                "content": reconciled_text,
                "reasoning_content": reconciled_reason,
            }
            tmp_path.write_text(
                json.dumps(tmp, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    else:
        final_json = {**counts_json, **texts_json}

    if final_json["Incident window start frame"] == -1:
        final_json["Incident window start frame"] = 1
    else:
        final_json["Incident window start frame"] = int(
            Path(image_paths[final_json["Incident window start frame"]]).stem.split(
                "."
            )[0]
        )

    ordered_final = {
        "video": video_id,
        **final_json,
    }
    out_path = Path(args.output_json)
    out_path.write_text(
        json.dumps(ordered_final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("===== final_json =====")
    print(ordered_final)
    print(f"Saved JSON to {out_path.resolve()}")


if __name__ == "__main__":
    main()
