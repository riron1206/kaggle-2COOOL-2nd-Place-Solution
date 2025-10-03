#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind A/B CSV rater for driving videos (English UI).
- Choose any 2 CSVs under ./csv in the index page
- Blind A/B is randomized per video and stored per pair
- Results are stored under ./results/<pair_key>/ (mapping.json, votes.csv)
- Each video page: two cards (left/right). Click a card to choose; Tie button at bottom.
- Video autoplays (muted, playsinline). No comments field.
WSL2: open http://localhost:8000
"""
import csv
import datetime
import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

APP_PORT = int(os.environ.get("APP_PORT", "8000"))
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# Videos to evaluate discovered from data/*.mp4
def discover_video_ids() -> List[int]:
    if not os.path.isdir(DATA_DIR):
        return []
    vids: List[int] = []
    for name in os.listdir(DATA_DIR):
        if not name.lower().endswith(".mp4"):
            continue
        m = re.match(r"^(\d+)\.mp4$", name)
        if m:
            try:
                vids.append(int(m.group(1)))
            except Exception:
                pass
    return sorted(set(vids))


os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ---------- CSV discovery / active pair state ----------


def list_csv_files() -> List[str]:
    if not os.path.isdir(CSV_DIR):
        return []
    files = [f for f in os.listdir(CSV_DIR) if f.lower().endswith(".csv")]
    files.sort()
    return files


ACTIVE_PAIR_PATH = os.path.join(RESULTS_DIR, "active_pair.json")


def get_active_pair() -> Tuple[str, str]:
    files = list_csv_files()
    if len(files) < 2:
        raise RuntimeError(f"Only {len(files)} CSV files found under csv/. Need at least two.")
    if os.path.exists(ACTIVE_PAIR_PATH):
        try:
            with open(ACTIVE_PAIR_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            a, b = d.get("A"), d.get("B")
            if a in files and b in files and a != b:
                return a, b
        except Exception:
            pass
    # default to first two
    a, b = files[0], files[1]
    set_active_pair(a, b, reset=True)  # initialize mapping
    return a, b


def set_active_pair(a: str, b: str, reset: bool = False) -> None:
    files = list_csv_files()
    if a not in files or b not in files or a == b:
        raise RuntimeError("Invalid pair selection.")
    with open(ACTIVE_PAIR_PATH, "w", encoding="utf-8") as f:
        json.dump({"A": a, "B": b}, f, ensure_ascii=False, indent=2)
    if reset:
        reset_pair_state(a, b)  # clears mapping & votes for this pair, then re-init on demand


def pair_key(a: str, b: str) -> str:
    # Stable key regardless of order? No: order matters because left/right is blinded per-video anyway,
    # but our pair concept is unordered comparison. We'll canonicalize by sorting basenames.
    xs = sorted([a, b])
    safe = "__VS__".join([re.sub(r"[^A-Za-z0-9._-]+", "_", x) for x in xs])
    # Also include a short hash to avoid overly long paths collisions
    h = hashlib.sha1(("::".join(xs)).encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{h}"


def pair_dir(a: str, b: str) -> str:
    d = os.path.join(RESULTS_DIR, pair_key(a, b))
    os.makedirs(d, exist_ok=True)
    return d


def mapping_path(a: str, b: str) -> str:
    return os.path.join(pair_dir(a, b), "mapping.json")


def votes_path(a: str, b: str) -> str:
    return os.path.join(pair_dir(a, b), "votes.csv")


def reset_pair_state(a: str, b: str) -> None:
    """Delete mapping.json and votes.csv for the pair (if exist)."""
    for p in [mapping_path(a, b), votes_path(a, b)]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ---------- CSV loading ----------


def read_csv_to_dict(path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = None
            for k in row.keys():
                if k.strip().lower() == "video":
                    key = k
                    break
            if key is None:
                continue
            try:
                vid = int(row[key])
            except ValueError:
                continue
            normalized = {k.strip(): (v if v is not None else "") for k, v in row.items()}
            out[vid] = normalized
    return out


def load_pair_data(a: str, b: str) -> Dict[str, Dict[int, Dict[str, str]]]:
    data = {}
    for name in [a, b]:
        full = os.path.join(CSV_DIR, name)
        data[name] = read_csv_to_dict(full)
    return data


# ---------- Mapping / votes per pair ----------


def ensure_mapping_for_pair(a: str, b: str) -> Dict[str, Any]:
    mpath = mapping_path(a, b)
    basenames = sorted([a, b])
    video_ids = discover_video_ids()
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                m = json.load(f)
            if sorted(m.get("csv_files", [])) == basenames and set(
                m.get("assignments", {}).keys()
            ) == set(str(v) for v in video_ids):
                return m
        except Exception:
            pass
        # stale -> rebuild
        try:
            os.remove(mpath)
        except Exception:
            pass
    assignments = {}
    for vid in video_ids:
        if random.random() < 0.5:
            assignments[str(vid)] = {"A": basenames[0], "B": basenames[1]}
        else:
            assignments[str(vid)] = {"A": basenames[1], "B": basenames[0]}
    m = {"csv_files": basenames, "assignments": assignments}
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return m


def list_done_videos(a: str, b: str) -> List[int]:
    vpath = votes_path(a, b)
    done = set()
    if os.path.exists(vpath):
        with open(vpath, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    done.add(int(row.get("video", "-1")))
                except Exception:
                    pass
    return sorted(done)


def save_vote(a: str, b: str, vid: int, choice: str, assignment: Dict[str, str]) -> None:
    vpath = votes_path(a, b)
    file_exists = os.path.exists(vpath)
    with open(vpath, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "timestamp",
            "pair",
            "video",
            "choice",
            "winner_file",
            "A_source_file",
            "B_source_file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        winner_file = ""
        if choice == "A":
            winner_file = assignment["A"]
        elif choice == "B":
            winner_file = assignment["B"]
        elif choice == "T":
            winner_file = "TIE"
        w.writerow(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "pair": " | ".join(sorted([a, b])),
                "video": vid,
                "choice": choice,
                "winner_file": winner_file,
                "A_source_file": assignment["A"],
                "B_source_file": assignment["B"],
            }
        )


def compute_summary(a: str, b: str, filter_videos: Any = None) -> Dict[str, Any]:
    vpath = votes_path(a, b)
    basenames = sorted([a, b])
    summary = {
        "per_file": {basenames[0]: 0, basenames[1]: 0},
        "ties": 0,
        "total": 0,
        "details": [],
    }
    if not os.path.exists(vpath):
        return summary
    with open(vpath, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                vid_val = int(row.get("video", "-1"))
            except Exception:
                vid_val = -1
            if filter_videos is not None and vid_val not in filter_videos:
                continue
            summary["total"] += 1
            wf = row.get("winner_file", "")
            if wf == "TIE":
                summary["ties"] += 1
            else:
                if wf in summary["per_file"]:
                    summary["per_file"][wf] += 1
            summary["details"].append(row)
    return summary


# ---------- Sampling (random100 stable subsets) ----------


def stable_random100_sets(video_ids: List[int]) -> Dict[str, List[int]]:
    """Return disjoint 100-sized sets as {"set1": [...], "set2": [...], ...}.
    Deterministic for the same discovered video id list.
    """
    vids = list(sorted(video_ids))
    count = len(vids) // 100
    if count <= 0:
        return {}
    # Deterministic shuffle by hashing the id list
    seed_src = "random100:" + ",".join(str(v) for v in vids)
    seed = int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:16], 16)
    rnd = random.Random(seed)
    shuffled = vids[:]
    rnd.shuffle(shuffled)
    sets: Dict[str, List[int]] = {}
    for i in range(count):
        start = i * 100
        end = start + 100
        sets[f"set{i+1}"] = sorted(shuffled[start:end])
    return sets


# ---------- Templates ----------

INDEX_TMPL = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind A/B Rater — Index</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; background: #f6f7f9; }
  header { background: white; border-bottom: 1px solid #e5e7eb; padding: 12px 16px; position: sticky; top: 0; }
  main { max-width: 980px; margin: 24px auto; padding: 0 16px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .btn { padding: 8px 12px; border-radius: 10px; border: 1px solid #d1d5db; background: #111827; color: white; text-decoration: none; }
  .btn.secondary { background: white; color: #111827; }
  label { font-size: 13px; color: #374151; display:block; margin-bottom:4px; }
  select { width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 8px; }
  table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
  th, td { padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }
  tr:last-child td { border-bottom: none; }
  .badge { padding: 2px 8px; border-radius: 9999px; font-size: 12px; }
  .done { background: #dcfce7; color: #065f46; }
  .todo { background: #fee2e2; color: #991b1b; }
  .actions { display:flex; gap:8px; flex-wrap:wrap; }
  .warn { background:#b91c1c; border-color:#7f1d1d; }
</style>
</head>
<body>
<header>
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div><strong>Blind A/B Rater</strong></div>
    <nav class="actions">
      <a class="btn secondary" href="{{ url_for('summary') }}">Summary</a>
    </nav>
  </div>
</header>
<main>
  {% with messages = get_flashed_messages() %}
    {% if messages %}<div style="color:#b91c1c; margin-bottom:8px; font-weight:600;">{{ messages[0] }}</div>{% endif %}
  {% endwith %}

  <div class="card">
    <h3>Choose two CSVs</h3>
    <form method="get" action="{{ url_for('index') }}" style="margin-bottom:12px;">
      <div class="row">
        <div>
          <label>Sampling</label>
          <div style="display:flex; gap:8px; align-items:center;">
            <select name="sample">
              <option value="">All videos</option>
              <option value="random100" {% if sample == 'random100' %}selected{% endif %}>random100</option>
            </select>
            {% if sample == 'random100' %}
            <select name="set">
              {% for s in random_sets %}
                <option value="{{ s }}" {% if set_name == s %}selected{% endif %}>{{ s }}</option>
              {% endfor %}
            </select>
            {% endif %}
            <button class="btn secondary" type="submit">Apply sampling</button>
          </div>
        </div>
      </div>
    </form>

    <form method="post" action="{{ url_for('set_pair_route') }}">
      <div class="row">
        <div>
          <label>CSV #1</label>
          <select name="csv1" required>
            {% for f in csv_files %}
              <option value="{{ f }}" {% if f == pair[0] %}selected{% endif %}>{{ f }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>CSV #2</label>
          <select name="csv2" required>
            {% for f in csv_files %}
              <option value="{{ f }}" {% if f == pair[1] %}selected{% endif %}>{{ f }}</option>
            {% endfor %}
          </select>
        </div>
      </div>
      <div class="actions" style="margin-top:12px;">
        <button class="btn" type="submit" name="mode" value="apply">Apply</button>
        <button class="btn warn" type="submit" name="mode" value="reset">Reset & Restart</button>
        {% if sample and set_name %}
          <a class="btn secondary" href="{{ url_for('video_page', vid=start_vid) }}?sample={{ sample }}&set={{ set_name }}">Start ▶</a>
        {% else %}
          <a class="btn secondary" href="{{ url_for('video_page', vid=start_vid) }}">Start ▶</a>
        {% endif %}
      </div>
      <div style="margin-top:8px; font-size:12px; color:#6b7280;">
        Active pair directory: <code>results/{{ pair_key }}</code>
      </div>
    </form>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Video progress</h3>
    <table>
      <thead><tr><th>Video</th><th>Status</th><th>Open</th></tr></thead>
      <tbody>
      {% for vid in vids %}
        <tr>
          <td>{{ vid }}.mp4</td>
          <td>
            {% if vid in done %}
              <span class="badge done">Done</span>
            {% else %}
              <span class="badge todo">Pending</span>
            {% endif %}
          </td>
          <td>
            {% if sample and set_name %}
              <a class="btn secondary" href="{{ url_for('video_page', vid=vid) }}?sample={{ sample }}&set={{ set_name }}">Open</a>
            {% else %}
              <a class="btn secondary" href="{{ url_for('video_page', vid=vid) }}">Open</a>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""

PAGE_TMPL = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video {{ vid }} — Evaluate</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; background: #f6f7f9; }
  header { background: white; border-bottom: 1px solid #e5e7eb; padding: 12px 16px; position: sticky; top: 0; }
  main { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); cursor: pointer; }
  .label { font-size: 12px; color: #6b7280; margin-bottom: 6px; }
  .value { font-size: 14px; color: #111827; white-space: pre-wrap; }
  .video { width: 100%; border-radius: 12px; overflow: hidden; background: #000; }
  .meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }
  .btnrow { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
  .btn { padding: 10px 14px; border-radius: 10px; border: 1px solid #d1d5db; background: #111827; color: white; text-decoration: none; }
  .btn.secondary { background: white; color: #111827; }
  .progress { font-size: 13px; color: #6b7280; }
  .flash { color: #b91c1c; font-weight: 600; }
</style>
<script>
  function choose(val){
    const f = document.getElementById('voteForm');
    f.choice.value = val;
    f.submit();
  }
</script>
</head>
<body>
<header>
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div><strong>Blind A/B</strong> <span class="progress">progress {{ done }}/{{ total }}</span></div>
    <nav style="display:flex; gap:8px;">
      {% if sample and set_name %}
        <a class="btn secondary" href="{{ url_for('index') }}?sample={{ sample }}&set={{ set_name }}">Index</a>
        <a class="btn secondary" href="{{ url_for('summary') }}?sample={{ sample }}&set={{ set_name }}">Summary</a>
      {% else %}
        <a class="btn secondary" href="{{ url_for('index') }}">Index</a>
        <a class="btn secondary" href="{{ url_for('summary') }}">Summary</a>
      {% endif %}
    </nav>
  </div>
</header>
<main>
  {% with messages = get_flashed_messages() %}
    {% if messages %}<div class="flash">{{ messages[0] }}</div>{% endif %}
  {% endwith %}

  <div class="card" style="cursor:default;">
    <div class="label">Video</div>
    <video class="video" src="{{ url_for('serve_video', vid=vid) }}" controls autoplay muted playsinline preload="metadata"></video>
  </div>

  {% if sample and set_name %}
  <form id="voteForm" method="post" action="{{ url_for('video_page', vid=vid) }}?sample={{ sample }}&set={{ set_name }}">
  {% else %}
  <form id="voteForm" method="post" action="{{ url_for('video_page', vid=vid) }}">
  {% endif %}
    <input type="hidden" name="choice" value="">
    <div class="grid">
      <div class="card" onclick="choose('A')" title="Choose this">
        <div class="meta">
          <div><div class="label">Incident Detection</div><div class="value">{{ rowA.get('Incident Detection', '') }}</div></div>
          <div><div class="label">Crash Severity</div><div class="value">{{ rowA.get('Crash Severity', '') }}</div></div>
          <div><div class="label">Ego-car involved</div><div class="value">{{ rowA.get('Ego-car involved', '') }}</div></div>
        </div>
        <div style="margin-top:10px;">
          <div class="label">Caption Before Incident</div>
          <div class="value">{{ rowA.get('Caption Before Incident', '') }}</div>
        </div>
        <div style="margin-top:10px;">
          <div class="label">Reason of Incident</div>
          <div class="value">{{ rowA.get('Reason of Incident', '') }}</div>
        </div>
      </div>

      <div class="card" onclick="choose('B')" title="Choose this">
        <div class="meta">
          <div><div class="label">Incident Detection</div><div class="value">{{ rowB.get('Incident Detection', '') }}</div></div>
          <div><div class="label">Crash Severity</div><div class="value">{{ rowB.get('Crash Severity', '') }}</div></div>
          <div><div class="label">Ego-car involved</div><div class="value">{{ rowB.get('Ego-car involved', '') }}</div></div>
        </div>
        <div style="margin-top:10px;">
          <div class="label">Caption Before Incident</div>
          <div class="value">{{ rowB.get('Caption Before Incident', '') }}</div>
        </div>
        <div style="margin-top:10px;">
          <div class="label">Reason of Incident</div>
          <div class="value">{{ rowB.get('Reason of Incident', '') }}</div>
        </div>
      </div>
    </div>

    <div class="btnrow">
      <button class="btn" type="button" onclick="choose('T')">Tie</button>
    </div>
  </form>
</main>
</body>
</html>
"""

SUMMARY_TMPL = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind A/B — Summary</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; background: #f6f7f9; }
  header { background: white; border-bottom: 1px solid #e5e7eb; padding: 12px 16px; position: sticky; top: 0; }
  main { max-width: 1000px; margin: 24px auto; padding: 0 16px; }
  .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
  th, td { padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }
  tr:last-child td { border-bottom: none; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
  .btn { padding: 8px 12px; border-radius: 10px; border: 1px solid #d1d5db; background: #111827; color: white; text-decoration: none; }
  .btn.secondary { background: white; color: #111827; }
</style>
</head>
<body>
<header>
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div><strong>Blind A/B — Summary</strong></div>
    <nav style="display:flex; gap:8px;">
      {% if sample and set_name %}
        <a class="btn secondary" href="{{ url_for('index') }}?sample={{ sample }}&set={{ set_name }}">Index</a>
      {% else %}
        <a class="btn secondary" href="{{ url_for('index') }}">Index</a>
      {% endif %}
    </nav>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card">
      <div>Total votes</div>
      <div style="font-size:28px; font-weight:700;">{{ summary.total }}</div>
    </div>
    <div class="card">
      <div>Ties</div>
      <div style="font-size:28px; font-weight:700;">{{ summary.ties }}</div>
    </div>
    <div class="card">
      <div>Winner</div>
      {% set a = file_order[0] %}
      {% set b = file_order[1] %}
      {% if summary.per_file[a] > summary.per_file[b] %}
        <div style="font-size:20px; font-weight:700;">{{ a }}</div>
        <div><span class="pill">{{ summary.per_file[a] }} vs {{ summary.per_file[b] }}</span></div>
      {% elif summary.per_file[a] < summary.per_file[b] %}
        <div style="font-size:20px; font-weight:700;">{{ b }}</div>
        <div><span class="pill">{{ summary.per_file[a] }} vs {{ summary.per_file[b] }}</span></div>
      {% else %}
        <div style="font-size:20px; font-weight:700;">Tie</div>
        <div><span class="pill">{{ summary.per_file[a] }} vs {{ summary.per_file[b] }}</span></div>
      {% endif %}
    </div>
  </div>

  <div class="card">
    <h3>Per-video A/B assignment (revealed)</h3>
    <p>Assignments are randomized per video for the active pair.</p>
    <table>
      <thead><tr><th>Video</th><th>Left (A)</th><th>Right (B)</th></tr></thead>
      <tbody>
        {% for vid in vids %}
        <tr><td>{{ vid }}</td><td>{{ mapping['assignments'][vid|string]['A'] }}</td><td>{{ mapping['assignments'][vid|string]['B'] }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h3>Votes (download)</h3>
    <p>
      {% if sample and set_name %}
        <a class="btn" href="{{ url_for('download_votes') }}">Download votes.csv (all)</a>
      {% else %}
        <a class="btn" href="{{ url_for('download_votes') }}">Download votes.csv</a>
      {% endif %}
    </p>
  </div>
</main>
</body>
</html>
"""

# ---------- Routes ----------


@app.route("/", methods=["GET"])
def index():
    csv_files = list_csv_files()
    try:
        a, b = get_active_pair()
    except RuntimeError as e:
        return f"<pre>{e}</pre>", 500

    m = ensure_mapping_for_pair(a, b)
    data = load_pair_data(a, b)
    done = set(list_done_videos(a, b))
    all_video_ids = sorted(int(k) for k in m.get("assignments", {}).keys())

    # subset selection via query
    sample = request.args.get("sample", "").strip()
    set_name = request.args.get("set", "").strip()
    random_sets = stable_random100_sets(all_video_ids)
    selected_videos = all_video_ids
    if sample == "random100" and set_name in random_sets:
        selected_videos = sorted(random_sets[set_name])
    elif sample == "random100" and not random_sets:
        flash("random100は動画が100本以上で利用できます。")

    video_ids = selected_videos
    # find first incomplete
    start_vid = None
    for vid in video_ids:
        if vid not in done:
            start_vid = vid
            break
    # 追加: フォールバック（一覧はあるが全部Doneの場合など）
    if start_vid is None:
        start_vid = video_ids[0] if video_ids else None

    return render_template_string(
        INDEX_TMPL,
        csv_files=csv_files,
        pair=(a, b),
        pair_key=pair_key(a, b),
        vids=video_ids,
        done=done,
        start_vid=start_vid,
        sample=sample,
        set_name=set_name,
        random_sets=list(sorted(random_sets.keys())),
    )


@app.route("/set_pair", methods=["POST"])
def set_pair_route():
    csv1 = request.form.get("csv1", "").strip()
    csv2 = request.form.get("csv2", "").strip()
    mode = request.form.get("mode", "apply")
    files = list_csv_files()
    if csv1 not in files or csv2 not in files or csv1 == csv2:
        flash("Invalid pair selection.")
        return redirect(url_for("index"))
    if mode == "reset":
        # set pair and reset its state
        set_active_pair(csv1, csv2, reset=True)
        flash(f"Pair set to [{csv1}] vs [{csv2}] and state RESET.")
    else:
        # set pair; do not destroy existing votes unless it is a new pair (mapping will be created if missing)
        set_active_pair(csv1, csv2, reset=False)
        flash(f"Pair set to [{csv1}] vs [{csv2}].")
    return redirect(url_for("index"))


@app.route("/video/<int:vid>", methods=["GET", "POST"])
def video_page(vid: int):
    a, b = get_active_pair()
    mapping = ensure_mapping_for_pair(a, b)
    all_video_ids = sorted(int(k) for k in mapping.get("assignments", {}).keys())

    # subset via query
    sample = request.args.get("sample", "").strip()
    set_name = request.args.get("set", "").strip()
    if sample == "random100":
        sets = stable_random100_sets(all_video_ids)
        video_ids = sorted(sets.get(set_name, [])) if set_name in sets else all_video_ids
    else:
        video_ids = all_video_ids
    if vid not in video_ids:
        abort(404)
    data = load_pair_data(a, b)

    assignment = mapping["assignments"][str(vid)]
    rowA = data.get(assignment["A"], {}).get(vid, {})
    rowB = data.get(assignment["B"], {}).get(vid, {})

    if request.method == "POST":
        choice = request.form.get("choice", "").strip()
        if choice not in ("A", "B", "T"):
            flash("No selection was made.")
        else:
            save_vote(a, b, vid, choice, assignment)
            done = set(list_done_videos(a, b))
            next_vid = None
            for nxt in video_ids:
                if nxt > vid and nxt not in done:
                    next_vid = nxt
                    break
            if next_vid is None:
                for nxt in video_ids:
                    if nxt not in done:
                        next_vid = nxt
                        break
            if next_vid is None:
                if sample == "random100" and set_name:
                    return redirect(url_for("summary") + f"?sample={sample}&set={set_name}")
                return redirect(url_for("summary"))
            # keep query in navigation
            if sample == "random100" and set_name:
                return redirect(url_for("video_page", vid=next_vid) + f"?sample={sample}&set={set_name}")
            return redirect(url_for("video_page", vid=next_vid))

    done_count = len(list_done_videos(a, b))
    return render_template_string(
        PAGE_TMPL,
        vid=vid,
        rowA=rowA,
        rowB=rowB,
        total=len(video_ids),
        done=done_count,
        sample=sample,
        set_name=set_name,
    )


@app.route("/data/<int:vid>.mp4")
def serve_video(vid: int):
    path = os.path.join(DATA_DIR, f"{vid}.mp4")
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(DATA_DIR, f"{vid}.mp4", as_attachment=False, mimetype="video/mp4")


@app.route("/results/votes.csv")
def download_votes():
    a, b = get_active_pair()
    vpath = votes_path(a, b)
    if not os.path.exists(vpath):
        # create an empty file with header so user can download something meaningful
        with open(vpath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp",
                    "pair",
                    "video",
                    "choice",
                    "winner_file",
                    "A_source_file",
                    "B_source_file",
                ]
            )
    return send_from_directory(
        os.path.dirname(vpath), os.path.basename(vpath), as_attachment=True, mimetype="text/csv"
    )


@app.route("/summary")
def summary():
    a, b = get_active_pair()
    mapping = ensure_mapping_for_pair(a, b)
    # subset via query
    all_video_ids = sorted(int(k) for k in mapping.get("assignments", {}).keys())
    sample = request.args.get("sample", "").strip()
    set_name = request.args.get("set", "").strip()
    filter_videos = None
    if sample == "random100":
        sets = stable_random100_sets(all_video_ids)
        if set_name in sets:
            filter_videos = set(sets[set_name])
    s = compute_summary(a, b, filter_videos=filter_videos)
    file_order = sorted([a, b])
    return render_template_string(
        SUMMARY_TMPL,
        summary=s,
        mapping=mapping,
        file_order=file_order,
        vids=(sorted(filter_videos) if filter_videos is not None else all_video_ids),
        sample=sample,
        set_name=set_name,
    )


if __name__ == "__main__":
    print(f" * CSV_DIR={CSV_DIR}")
    print(f" * DATA_DIR={DATA_DIR}")
    print(f" * RESULTS_DIR={RESULTS_DIR}")
    print(f" * Access from Windows: http://localhost:{APP_PORT}")
    app.run(host=APP_HOST, port=APP_PORT, debug=False)
