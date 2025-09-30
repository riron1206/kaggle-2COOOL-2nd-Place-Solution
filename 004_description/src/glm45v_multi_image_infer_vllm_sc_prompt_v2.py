import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from collections import Counter
from statistics import median


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
You are a traffic-incident annotator for multi-frame dashcam inputs (earliest → latest). Use temporal changes across frames.

Reason silently in **ULTRATHINK** mode, then return STRICT JSON.

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
- Index of the image where the hazard/accident first becomes visibly evident.
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
  0 = Not involved (no collision/physical interaction)
  1 = Involved (collides, is hit, or physically interacts)

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
- Bicyclists/Scooters = cyclists or scooters directly involved.
- Animals = animals directly involved (collision or blocking the lane).
- Pedestrians = pedestrians directly involved (in path/impact/clear near-miss).
- Vehicles involved (excluding ego-car) = directly involved motor vehicles.

Decision guidelines:
- Incident Detection = 1 if any visible collision occurs.
- Use 0 for non-contact anomalies (debris, animal, flooded road, downed power line, fog/smoke, fire) or clear near-miss.
- Map Crash Severity consistently with observed involvement. If no collision, use "0. No Crash".

Style for downstream caption stability:
- Prefer canonical tokens used in traffic datasets: ego-car, car, truck, bus, van, motorcycle, bicycle, pedestrian, lane, crosswalk, intersection, oncoming traffic, traffic light (red/green/yellow), debris, wet road, fog, smoke, fire.
- Avoid speculation or rare synonyms ("automobile", "motorbike", "appears", "likely").
""".strip()

PROMPT_TEXT: str = """
You are a traffic-scene captioner evaluated by CIDEr-D, METEOR, and SPICE.
Input consists of consecutive dashcam frames (earliest → latest). Exploit temporal cues.

Work in **ULTRATHINK** mode (think silently). Then output STRICT JSON.

Writing checklist to maximize CIDEr-D / METEOR / SPICE:
- Exactly ONE sentence per field; 12–16 words; present tense; active voice; fluent English.
- Include at least: (a) concrete objects, (b) one attribute (color/number/state), (c) one spatial relation.
- Use canonical tokens only: ego-car, car, truck, bus, van, motorcycle, bicycle, pedestrian, crosswalk, intersection, lane, oncoming traffic, traffic light (red/green/yellow), debris, wet road, fog, smoke, fire.
- Use articles ("a", "the"). Use digits for counts (e.g., 2 pedestrians, 3 cars) when visible.
- Name concrete actions: stops, brakes, merges, turns left/right, changes lane, overtakes, crosses, collides, blocks.
- Avoid hedging and rare synonyms (no "likely", "appears", "automobile", "motorbike").
- Do not mention frame indices inside sentences.

Output rules (STRICT):
- Return pure JSON only with exactly these three keys.

JSON format:
{
  "Incident window start frame": <int: -1..N-1>,
  "Caption Before Incident": "<one-sentence objective description of the last stable scene before any anomaly>",
  "Reason of Incident": "<one-sentence concrete visual cause-and-effect; if unclear, write 'unknown hazard: <visible evidence>'>"
}

Definitions:
- "Incident window start frame": index where the hazard/accident first becomes visibly evident; -1 if none.
- "Caption Before Incident": stable, pre-incident scene (e.g., “The ego-car follows a white car in the right lane in city traffic.”).
- "Reason of Incident": explicit cause-and-effect with normalized terms and relations (e.g., “A car from the left lane cuts in ahead and forces hard braking without contact.”).

Style examples (for style only; adapt to the images, do not copy words):
{"Incident window start frame": 2,
 "Caption Before Incident": "The ego-car drives in the right lane on a city street near a crosswalk.",
 "Reason of Incident": "A pedestrian steps onto the crosswalk ahead from the right and blocks the ego-car lane."}

{"Incident window start frame": 1,
 "Caption Before Incident": "The ego-car follows a white car on a wet highway with three visible lanes.",
 "Reason of Incident": "A car from the left lane cuts in sharply ahead and causes a near-miss without contact."}
""".strip()

PROMPT_RECONCILE: str = """
You are an assistant that reconciles two prior predictions using the images as ground truth.

Work in **ULTRATHINK** mode (think silently).

Inputs:
- Images: consecutive frames from a single dashcam video (earliest → latest). Use temporal evidence.
- First prediction: Counts JSON.
- Second prediction: Texts JSON.

Goal:
- Produce ONE self-consistent JSON with EXACTLY the 11 keys below, corrected by image evidence whenever the two predictions contradict.

Return format (STRICT):
- Return pure JSON only — no code fences or extra text.
- Use EXACTLY these 11 keys (spelling must match):
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
- Numeric fields must be integers. Text fields must use allowed sets below (except the two caption fields).
- Base everything on visible evidence in the images. Do not speculate.

Allowed values (reuse these EXACT strings):

Incident window start frame (Numeric):
  Index where the hazard/accident first becomes visibly evident; -1 if no anomaly.

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

Style for the two text fields (CIDEr/METEOR/SPICE friendly):
- "Caption Before Incident": ONE concise sentence (12–16 words), present tense, objective, pre-incident only.
- "Reason of Incident": ONE concise sentence (12–16 words) explaining visible cause-and-effect using canonical tokens and at least one spatial relation (left/right/ahead/behind/in/on/at/near). If unchanged from the Texts JSON, keep that value.

Evidence precedence when predictions contradict:
Images > Texts JSON > Counts JSON.

Global consistency rules (MUST satisfy all that apply):
A) If "Incident Detection" = -1:
   - "Crash Severity" = "0. No Crash"
   - "Ego-car involved" = 0
   - "Label" = "unknown"
   - All counts = 0
B) If "Incident Detection" = 0 (Hazard, no collision):
   - "Crash Severity" = "0. No Crash"
   - Default "Ego-car involved" = 0 (unless images show direct physical interaction).
   - Choose a hazard-compatible "Label":
     {"animal on the road","bicycle on road","scooter on the road",
      "pedestrian is crossing the street","pedestrian on the road",
      "vehicle overtakes","unknown"}.
C) If "Incident Detection" = 1 (Accident):
   - Set "Ego-car involved" from images.
   - If "Ego-car involved" = 1:
       * Collision with person/cyclist → "3. Ego-car collided with at-least one person or cyclist".
       * Ego continues to move → "1. Ego-car collided but did not stop".
       * Ego cannot continue → "2. Ego-car collided and could not continue moving".
       * "Label" ∈ {
         "ego-car hits barrier","ego-car hits a pedestrian","ego-car hits a crossing cyclist",
         "ego-car hits a vehicle","vehicle hits ego-car","ego-car hit an animal",
         "flying object hit the car","ego-car loses control"}.
   - If "Ego-car involved" = 0:
       * Other vehicles collide with a person/cyclist/object → "4. Other cars collided with person/car/object but ego-car is ok".
       * Vehicle↔vehicle only → "6. One or Multiple vehicles collided but ego-car is fine".
       * "Label" should match images; common choices:
         {"vehicle drives into another vehicle","car hits barrier","car flipped over",
          "many cars/pedestrians/cyclists collided","unknown"}.

Counting alignment rules:
- Count ONLY directly involved participants (exclude bystanders).
- If "Reason of Incident" mentions "pedestrian/cyclist/animal", set corresponding count ≥ 1 (if visible).
- If "Label" = "vehicle drives into another vehicle" and ego is not involved, set "Number of vehicles involved (excluding ego-car)" ≥ 2.
- If "Label" mentions ego-car interactions, ensure "Ego-car involved" = 1.

Sanity checks:
- "Incident Detection" = 1 ⇒ "Crash Severity" ≠ "0. No Crash".
- "Incident Detection" ∈ {0, -1} ⇒ "Crash Severity" = "0. No Crash".
- Any count > 0 ⇒ "Incident Detection" ∈ {0,1}.
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
    Check for simple-rule contradictions between counts_json (8 numeric keys)
    and texts_json (2 text keys). Return a list of messages (empty if none).
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
    print("[Debug] output tokens:", resp.usage.completion_tokens)

    content: str = ""
    reasoning_content: str = ""
    try:
        content = resp.choices[0].message.content or ""
        if "</think>" in content:
            idx = content.find("</think>")
            reasoning_content = content[:idx]
            content = content[idx + len("</think>") :]
    except Exception:
        content = ""

    try:
        rc = getattr(resp.choices[0].message, "reasoning_content", None)
        if rc and not reasoning_content:
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


def _majority_vote(values, prefer=None, ignore=None):
    """Return the mode; tie-broken by numeric ascending or lexical order.
    `ignore` values are dropped. `prefer` (list) can be used to prioritize candidates.
    """
    vals = [v for v in values if (ignore is None or v not in ignore)]
    if not vals:
        return None
    c = Counter(vals)
    maxc = max(c.values())
    winners = [k for k, v in c.items() if v == maxc]
    if prefer:
        for p in prefer:
            if p in winners:
                return p
    try:
        return sorted(winners)[0]
    except Exception:
        return winners[0]


def _median_int(values):
    vals = [int(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return 0
    try:
        return int(round(median(vals)))
    except Exception:
        vals.sort()
        mid = len(vals) // 2
        return vals[mid]


def chat_with_images_multi(
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
    n_samples: int,
):
    """Return (contents_list, reasoning_list). Uses one API call with n choices when possible."""
    client = OpenAI(base_url=api_base, api_key=api_key)
    image_parts = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, *image_parts]}
    ]

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
        n=n_samples,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["extra_body"] = {
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    }

    resp = client.chat.completions.create(**kwargs)
    contents, reasonings = [], []
    for i, ch in enumerate(getattr(resp, "choices", []) or []):
        msg = getattr(ch, "message", None)
        content = ""
        reasoning_content = ""
        if msg is not None:
            content = getattr(msg, "content", "") or ""
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                reasoning_content = rc
        if not reasoning_content:
            rc2 = getattr(ch, "reasoning_content", None)
            if rc2:
                reasoning_content = rc2
        contents.append(content)
        reasonings.append(reasoning_content)

    print(f"[SelfConsistency] got {len(contents)} samples (requested n={n_samples})")
    for idx, c in enumerate(contents[:3]):
        print(f"--- sample[{idx}] content head ---\n{c[:1000]}\n")
    return contents, reasonings


def aggregate_final_jsons(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate reconciled JSONs (11 keys expected, but works with partial keys)."""
    if not samples:
        return {}
    keys = set()
    for s in samples:
        keys.update(s.keys())
    agg = {}

    def collect(k):
        return [s.get(k) for s in samples if k in s]

    if "Incident Detection" in keys:
        agg["Incident Detection"] = _majority_vote(collect("Incident Detection"))
    ego_key = (
        "Ego-car involved"
        if "Ego-car involved" in keys
        else ("Ego car involved" if "Ego car involved" in keys else None)
    )
    if ego_key:
        agg["Ego-car involved"] = _majority_vote(collect(ego_key))
    if "Crash Severity" in keys:
        agg["Crash Severity"] = _majority_vote(collect("Crash Severity"))
    if "Label" in keys:
        agg["Label"] = _majority_vote(collect("Label"))

    for ck in [
        "Number of Bicyclists/Scooters",
        "Number of animals involved",
        "Number of pedestrians involved",
        "Number of vehicles involved (excluding ego-car)",
        "Number of Bicyclists Scooters",
        "Number of Animals",
        "Number of Pedestrians",
        "Number of Vehicles excluding ego",
    ]:
        if ck in keys:
            agg[ck] = _median_int(collect(ck))

    if "Incident window start frame" in keys:
        vals = [
            v
            for v in collect("Incident window start frame")
            if isinstance(v, (int, float))
        ]
        cand = _majority_vote(vals, ignore=[-1])
        if cand is None:
            if all(int(v) == -1 for v in vals) if vals else False:
                cand = -1
            else:
                cand = 0
        agg["Incident window start frame"] = int(cand)

    def score_text_sample(s):
        sc = 0
        for k in ["Incident Detection", "Crash Severity", "Label", "Ego-car involved"]:
            if k in agg and k in s and str(agg[k]) == str(s[k]):
                sc += 1
        return sc

    best = max(samples, key=score_text_sample)
    for tk in ["Caption Before Incident", "Reason of Incident"]:
        if tk in best:
            agg[tk] = best[tk]

    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        nargs="+",
        default=INPUT_IMAGES,
        help="Specify multiple input image paths (.jpg/.png etc.)",
    )
    ap.add_argument(
        "--reconcile",
        action="store_true",
        help="After two-stage inference (PROMPT_COUNT then PROMPT_TEXT), run reconciliation if they disagree",
    )
    ap.add_argument(
        "--tmp_json",
        default=None,
        help="Path to save intermediate JSON (counts/texts/reconciled plus content/reasoning_content)",
    )
    ap.add_argument("--output_json", default=OUTPUT_JSON, help="Output JSON path")
    ap.add_argument(
        "--api_base",
        default=DEFAULT_API_BASE,
        help="Base URL of vLLM OpenAI-compatible API (e.g., http://localhost:8000/v1)",
    )
    ap.add_argument(
        "--api_key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API key (use 'EMPTY' if not required)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Match the served model name (e.g., glm-4.5v)",
    )
    ap.add_argument(
        "--send_mode",
        choices=["file_url", "base64"],
        default="file_url",
        help="How to pass images: file_url= file:///... or base64= data:image/...;base64,...",
    )
    ap.add_argument("--max_tokens", type=int, default=8192, help="Maximum generated tokens")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 to reduce repetition)",
    )
    ap.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling")
    ap.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    ap.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for self-consistency (e.g., 8–16)"
    )
    ap.add_argument(
        "--sc_mode",
        choices=["majority", "weighted"],
        default="majority",
        help="Aggregation mode: majority or weighted (experimental)",
    )
    ap.add_argument(
        "--sc_at",
        choices=["final", "all"],
        default="final",
        help="Where to apply self-consistency: final=reconcile only / all=each stage + reconcile",
    )
    ap.add_argument(
        "--sc_temp",
        type=float,
        default=None,
        help="Override temperature only during self-consistency (default: normal temperature)",
    )
    ap.add_argument(
        "--sc_top_p", type=float, default=None, help="Override top_p only during self-consistency"
    )
    ap.add_argument(
        "--sc_top_k", type=int, default=None, help="Override top_k only during self-consistency"
    )
    ap.add_argument(
        "--timeout", type=float, default=600.0, help="HTTP request timeout seconds"
    )
    ap.add_argument(
        "--json_mode",
        action="store_true",
        help="Enable response_format={'type':'json_object'}",
    )
    ap.add_argument(
        "--counts_only",
        action="store_true",
        help="Run Counts inference only and output Counts keys then exit",
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

    if args.sc_at == "all" and args.n_samples > 1:
        sc_temp = args.sc_temp if args.sc_temp is not None else args.temperature
        sc_top_p = args.sc_top_p if args.sc_top_p is not None else args.top_p
        sc_top_k = args.sc_top_k if args.sc_top_k is not None else args.top_k
        contents, reasonings = chat_with_images_multi(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            image_urls=image_urls,
            prompt=PROMPT_COUNT,
            json_mode=args.json_mode,
            request_timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=sc_temp,
            repetition_penalty=args.repetition_penalty,
            top_p=sc_top_p,
            top_k=sc_top_k,
            n_samples=args.n_samples,
        )
        jsons = [
            post_validate(parse_json_robust(c), expected_keys=COUNT_KEYS)
            for c in contents
        ]
        counts_json = aggregate_final_jsons(jsons)
        counts_text = contents[0] if contents else ""
        counts_reason = reasonings[0] if reasonings else ""
    else:
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

    if args.sc_at == "all" and args.n_samples > 1:
        sc_temp = args.sc_temp if args.sc_temp is not None else args.temperature
        sc_top_p = args.sc_top_p if args.sc_top_p is not None else args.top_p
        sc_top_k = args.sc_top_k if args.sc_top_k is not None else args.top_k
        contents, reasonings = chat_with_images_multi(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            image_urls=image_urls,
            prompt=PROMPT_TEXT,
            json_mode=args.json_mode,
            request_timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=sc_temp,
            repetition_penalty=args.repetition_penalty,
            top_p=sc_top_p,
            top_k=sc_top_k,
            n_samples=args.n_samples,
        )
        jsons = [
            post_validate(parse_json_robust(c), expected_keys=TEXT_KEYS)
            for c in contents
        ]
        texts_json = aggregate_final_jsons(jsons)
        texts_text = contents[0] if contents else ""
        texts_reason = reasonings[0] if reasonings else ""
    else:
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
        if args.n_samples > 1:
            sc_temp = args.sc_temp if args.sc_temp is not None else args.temperature
            sc_top_p = args.sc_top_p if args.sc_top_p is not None else args.top_p
            sc_top_k = args.sc_top_k if args.sc_top_k is not None else args.top_k
            contents, reasonings = chat_with_images_multi(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                image_urls=image_urls,
                prompt=reconcile_prompt,
                json_mode=args.json_mode,
                request_timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=sc_temp,
                repetition_penalty=args.repetition_penalty,
                top_p=sc_top_p,
                top_k=sc_top_k,
                n_samples=args.n_samples,
            )
            jsons = [
                post_validate(parse_json_robust(c), expected_keys=JSN_KEYS)
                for c in contents
            ]
            final_json = aggregate_final_jsons(jsons)
            if has_empty_values(final_json, JSN_KEYS):
                base_rp = args.repetition_penalty
                for i in range(RETRY_ON_EMPTY_MAX_ATTEMPTS):
                    new_rp = min(
                        base_rp + (i + 1) * RETRY_ON_EMPTY_STEP, RETRY_ON_EMPTY_CAP
                    )
                    contents, reasonings = chat_with_images_multi(
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=args.model,
                        image_urls=image_urls,
                        prompt=reconcile_prompt,
                        json_mode=args.json_mode,
                        request_timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        temperature=sc_temp,
                        repetition_penalty=new_rp,
                        top_p=sc_top_p,
                        top_k=sc_top_k,
                        n_samples=args.n_samples,
                    )
                    jsons = [
                        post_validate(parse_json_robust(c), expected_keys=JSN_KEYS)
                        for c in contents
                    ]
                    final_json = aggregate_final_jsons(jsons)
                    if not has_empty_values(final_json, JSN_KEYS):
                        break
            reconciled_text = contents[0] if contents else ""
            reconciled_reason = reasonings[0] if reasonings else ""
        else:
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

    ordered_keys = [
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

    ordered_final: Dict[str, Any] = {"video": video_id}
    for k in ordered_keys:
        if k in final_json:
            ordered_final[k] = final_json[k]
    out_path = Path(args.output_json)
    out_path.write_text(
        json.dumps(ordered_final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("===== final_json =====")
    print(ordered_final)
    print(f"Saved JSON to {out_path.resolve()}")


if __name__ == "__main__":
    main()
