# Roadscan — AI-Based Road Damage Assessment System

> An AI-based Road Damage Assessment System that uses YOLOv7 for road-damage
> detection and combines computer vision analysis with severity estimation,
> road-health scoring, repair prioritization, historical monitoring,
> before/after comparison, and automated inspection reporting.

Built on top of [`sivakanth1/Detecting_Road_Damage`](https://github.com/sivakanth1/Detecting_Road_Damage).
Uploaded-image analysis only — no live camera, no dashcam streaming, no
real-time driver warnings, by design.

---

## 1. Project overview

Roadscan takes a single uploaded road photograph and produces a full
infrastructure-assessment report: what damage is present, how severe it is,
an overall 0–100 health score, a repair priority, and (optionally) how the
road has changed over repeated inspections or after a repair.

## 2. Problem statement

Manual road inspection is slow, subjective, and hard to track consistently
over time. A single photograph already contains enough visual information
to flag obvious surface damage (potholes, cracking) — this project turns
that observation into a structured, repeatable, explainable assessment
pipeline instead of a one-off "yes/no damage" classifier.

## 3. Objectives

1. Detect and classify multiple road-damage types in an uploaded image.
2. Localize each damage (bounding box) and approximate its extent (region/mask).
3. Score the severity of each individual damage from measurable features.
4. Aggregate into one overall, explainable Road Health Score (0–100).
5. Recommend a repair priority with a stated reason.
6. Track the same location over repeated inspections.
7. Compare before/after repair photos without overclaiming success.
8. Produce a professional, exportable report (PDF/HTML/JSON).

## 4. System architecture

```
                ROAD IMAGE
                    |
                    v
              YOLOv7 Detector            <- detector.py
        (falls back to classical CV
         if YOLOv7 isn't available,
         and says so explicitly)
                    |
                    v
            Damage Detection             <- Detection objects: class, bbox,
                    |                       confidence, region mask
                    v
        Region/Mask Visualization        <- explain.py
                    |
                    v
            Severity Assessment          <- severity.py
                    |
                    v
            Road Health Score            <- health_score.py
                    |
                    v
             Repair Priority             <- severity.py (compute_repair_priority)
                    |
          +---------+---------+
          v                   v
      SQLite History      Before/After
      (history.py)        (comparison.py)
          |                   |
          +---------+---------+
                    v
             Report Generator            <- report.py
                    |
             +------+------+
             v      v      v
            HTML    PDF    JSON
                    |
                    v
              Web Dashboard              <- app.py + templates/
```

## 5. Dataset

The detection classes (D00/D10/D20/D40) follow **RDD2022** (Road Damage
Dataset 2022), the same dataset used by the upstream project this repo
extends. This repository does **not** bundle the dataset or retrain
anything — see "Model/weight requirements" below for exactly what training
artifacts are (and are not) included.

## 6. Data preprocessing

At inference time, `app.py` resizes any uploaded image so its longest side
is ≤1600px (keeps inference fast on a normal laptop) and both detector
backends run their own internal preprocessing:
- `HeuristicCVDetector`: grayscale conversion, CLAHE contrast enhancement,
  Gaussian blur.
- `YoloV7EnsembleDetector`: YOLO-style letterbox resize + BGR→RGB + 0–255→0–1
  normalization, mirroring `index.ipynb`.

No offline dataset preprocessing pipeline is included in this repo — that
lives in the upstream project's training notebooks, which are unchanged.

## 7. YOLOv7 architecture

The **primary** detector is an ensemble of two custom-trained YOLOv7 models
(anchor-based, single-stage CNN detector) at two input resolutions —
640×640 and 512×512 — with their outputs merged via non-max suppression.
This mirrors the upstream project's `index.ipynb` exactly (same class order,
same two-resolution ensemble idea). See section 18 for exact activation
requirements.

**Important:** this sandbox has no GPU, no `torch`, and blocked network
access, so the YOLOv7 code path is implemented but has **not been executed
here**. `detector.get_default_detector()` tries YOLOv7 first and only
falls back to the classical-CV detector — clearly labeled wherever it's
used — if YOLOv7's dependencies aren't present. It never presents fallback
output as YOLOv7 output.

## 8. Detection pipeline

`detector.py` defines one interface, `RoadDamageDetector.detect(image) ->
List[Detection]`, with two implementations:

| | `YoloV7EnsembleDetector` (PRIMARY) | `HeuristicCVDetector` (FALLBACK) |
|---|---|---|
| What it is | Trained CNN ensemble | Classical CV (CLAHE + contour/edge/blob analysis) |
| Status | Requires torch + weights (see §18) | Runs today, no downloads |
| Output | class, bbox, confidence (no mask) | class, bbox, confidence, contour-fill mask |
| Accuracy | Whatever the trained model achieves | Illustrative baseline only — not benchmarked against YOLOv7 |

Every `Detection` has: `class_id`, `class_name`, `confidence`, `bbox`,
`mask` (optional), and a unique Python object identity used as its
detection ID within a single analysis run.

## 9. Segmentation / region visualization methodology

**Honesty note, as required:** neither backend currently produces a
pixel-perfect *learned* segmentation mask.
- `HeuristicCVDetector` fills the actual contour it detected — a real,
  image-derived region, tighter than a bounding box, but not a
  trained-segmentation-network output.
- `YoloV7EnsembleDetector` is a plain object detector (bounding boxes only,
  `mask=None`); a true YOLOv7-seg or separate segmentation head would be
  needed for learned masks and is not part of this build.

The UI and reports both say "approximate region (contour-based), not a
learned pixel-perfect segmentation mask" rather than overclaiming.

**Explainability** is a separate, clearly-labeled concept from both of the
above: `explain.evidence_heatmap()` produces a heatmap of which pixels
drove detection, built from the same edge/contrast signals the classical
detector used. **This is explicitly not Grad-CAM** — Grad-CAM requires
back-propagating gradients through a trained CNN's convolutional layers,
which only applies once the YOLOv7 backend is active with a differentiable
model available. The code/UI/report never label the current heatmap as
Grad-CAM.

## 10. Severity methodology (`severity.py`)

For each detection:

```
area_component = min(1, area_ratio / 0.08)
confidence_component = clamp((confidence - 0.25) / 0.75, 0, 1)
severity_score = 100 * (0.55*area_component + 0.25*confidence_component)
                       * (class_weight / 2.2)
                 + 8 * (class_weight - 1)
```

`class_weight`: D00/D10 = 1.0, D43 = 1.3, D20 = 1.6, D40 (pothole) = 2.2.

Label thresholds (configurable in `severity.SEVERITY_THRESHOLDS`):
`score < 25` → LOW, `< 50` → MODERATE, `< 75` → HIGH, else → CRITICAL.

Every assessment also carries a generated `reason` string, e.g.:

> "High severity: large affected area (4.2% of the image); pothole carries
> a high repair-risk weighting."

## 11. Health-score methodology (`health_score.py`)

```
damage_penalty = sum over detections of
    (severity_score/100) * class_weight * area_multiplier * 6.0
  where area_multiplier = 1 + min(1.5, area_ratio / 0.03)

count_penalty = min(20, 2.5 * max(0, damage_count - 1))

Road Health Score = 100 - (damage_penalty + count_penalty), clamped to [0, 100]
```

`penalty_by_class` re-aggregates `damage_penalty` per damage type so the UI
and reports can show, e.g.:

```
Road Health Score: 64/100
Penalty breakdown:
- Potholes: -16
- Longitudinal cracks: -10
- Alligator cracks: -10
```

Condition bands: 90–100 Excellent, 75–90 Good, 55–75 Fair, 35–55 Poor, 0–35 Severe.

## 12. Repair-priority methodology (`severity.compute_repair_priority`)

Deterministic rule cascade (evaluated top-down, first match wins):

- **URGENT** — a CRITICAL pothole, ≥2 CRITICAL-severity detections, or health score < 30
- **HIGH** — worst detection is CRITICAL, a large alligator-crack area, health score < 50, or ≥2 HIGH-or-CRITICAL detections
- **MEDIUM** — worst detection is MODERATE/HIGH, or health score < 75
- **LOW** — otherwise (including "no damage detected")

Every result includes `reason` (why this level) and `recommended_action`
(what to actually do).

## 13. Historical tracking (`history.py`, SQLite)

Each analysis, if given a location label, is stored in
`data/history.sqlite3` with timestamp, image reference, damage summary,
health score, and priority. `compare_latest_two(location)` compares the two
most recent records and returns `trend_status` — exactly one of
`"improving"`, `"stable"`, `"deteriorating"` — using an ≥8-point
health-score swing or a ≥2 change in damage count as the deciding rule.
**With fewer than two records for a location, no trend is computed or
displayed — the system never fabricates a trend from one data point.**
The location label is free text the user supplies; Roadscan does not use
GPS or image-matching to verify two uploads are the same physical spot.

## 14. Before/after comparison (`comparison.py`)

Runs the full pipeline on both images independently, then diffs health
score / damage count / affected area. Also computes a scene-similarity
score (HSV color-histogram correlation) between the two photos:

- similarity < 0.20 → verdict is **"Cannot reliably determine"**, with the
  message *"Comparison cannot reliably determine repair improvement because
  the scenes differ significantly."* — numeric deltas are still shown for
  reference but not interpreted as a repair outcome.
- 0.20 ≤ similarity < 0.35 → comparison proceeds but is flagged
  lower-confidence.
- similarity ≥ 0.35 → normal comparison (Improved / Remained similar /
  Appears worse) based on health-score and damage-count deltas.

## 15. Report generation (`report.py`)

Every analysis produces the same structured `report` dict, rendered three ways:

- **JSON** — machine-readable, matches the shape shown in your spec (`location`, `timestamp`, `health_score`, `priority`, `detections`, `severity_summary`-equivalent fields).
- **HTML** — a styled, self-contained, printable report (`templates/report.html`), viewable in-browser or downloadable.
- **PDF** — formatted like an inspection document (title, inspection ID, summary table, penalty breakdown, per-damage table with reasons, priority + recommendation, disclaimer) via ReportLab, not raw debug output.

Every report includes: title, inspection ID, timestamp, location, annotated
image, per-damage detail, Road Health Score + penalty breakdown, repair
priority + reason + recommendation, historical/before-after comparison
sections when available, and a decision-support disclaimer.

## 16. UI

Flask + Jinja2, asphalt/road-marking visual theme (`static/css/style.css`)
— summary cards (health score gauge, priority, damage count), annotated
image, penalty breakdown table, damage-type/severity distribution tables,
detection table with per-row reasons, explainability panel, history trend
chart (SVG line chart, no extra JS chart library), before/after page, and
PDF/HTML/JSON export buttons. A confidence-threshold slider on the upload
form filters detections post-inference. The active detector backend
(YOLOv7 vs fallback) is shown persistently in the footer, and a banner
appears site-wide whenever the fallback is active.

## 17. Installation

```bash
pip install -r requirements.txt
```

No GPU, model download, or dataset download is required to run the app as
shipped (fallback detector).

## 18. Running the application

```bash
python app.py
```

Open `http://localhost:5000`. Startup prints which detector backend is
active and why.

## 19. Model / weight requirements (YOLOv7 activation)

See `models/README.md` for full detail. Summary — all three required:

```
models/yolov7_repo/                 <- clone https://github.com/WongKinYiu/yolov7
models/weights/yolov7all6_best.pt   <- your 640x640-trained checkpoint
models/weights/yolov7all61_best.pt  <- your 512x512-trained checkpoint
```

Plus `pip install torch torchvision`. Without all three, Roadscan runs the
classical-CV fallback and says so everywhere — it never silently claims
heuristic output is YOLOv7 output.

## 20. Testing

- `tests/test_yolo_loading.py` — verifies the YOLOv7 loader once its
  dependencies exist. **Not executed in this sandbox** (no torch/GPU/network)
  — run it yourself after completing §19.
- `tests/run_pipeline_tests.py` — runs the real, currently-active pipeline
  against every image in `tests/images/` and writes actual results to
  `tests/test_results.md`. **As shipped, `tests/images/` is empty and no
  real-image results have been recorded** — fabricating them was avoided
  per project requirements. To get real coverage, add photos named:

  ```
  01_no_or_minimal_damage.jpg
  02_pothole.jpg
  03_longitudinal_crack.jpg
  04_transverse_crack.jpg
  05_multiple_damages.jpg
  06_severe_damage.jpg
  ```

  (daylight, road surface filling most of the frame, JPG/PNG, any
  resolution) into `tests/images/`, then run
  `python tests/run_pipeline_tests.py`. The full pipeline (detection through
  history/comparison) was exercised end-to-end via `app.test_client()`
  against a synthetic test image during development to confirm every route
  returns 200 and every template renders without error — that is a
  plumbing/integration check, not a substitute for real-image accuracy
  testing.

## 21. Limitations

- YOLOv7 integration is implemented but unverified in this environment (no
  torch/GPU/network here) — verify on your machine before trusting output.
- The fallback classical-CV detector is a baseline, not benchmarked against
  YOLOv7 accuracy; treat its output as illustrative.
- Segmentation is contour-fill, not a learned pixel-perfect mask.
- Severity/health-score weights are hand-set for plausibility, not fit to
  labeled ground truth (RDD2022 has no severity labels).
- History location-matching is user-declared text, not GPS/geo-verified.
- Before/after scene-similarity is a coarse histogram heuristic, not true
  scene/pose verification.
- No real-image test results are included yet (see §20).

## 22. Citizen Road Damage Reporting Module (`citizen_report.py`)

A civic-reporting workflow layered on top of the existing pipeline, taking
general UX inspiration from civic-issue-reporting apps (report → categorize
→ track → resolve) without copying any branding, logos, or screen designs
from any specific app.

**Workflow:** Home → *Report Road Damage* → upload photo + location
(+ optional description/road name/landmark/reporter info) → the photo is
run through the SAME `run_pipeline()` used by `/analyze` (no second
detector, no second severity/health/priority engine) → results are shown
→ on submit, a unique, persistent **Report ID** is generated
(`RS-<year>-<row id>`, reproducible from the stored row, never random) →
the citizen can **Track Report** by ID at any time, or see everything
they've submitted under **My Reports**.

**Status lifecycle:** `REPORTED → UNDER_REVIEW → ASSIGNED → IN_PROGRESS →
RESOLVED`, stored with a timestamped history for the timeline view. A
**Maintenance Dashboard** (`/maintenance`) lets a "staff" user update any
report's status; choosing *Resolved* offers an optional after-repair photo,
which is compared against the original using the **existing**
`comparison.py` before/after logic — no duplicate comparison engine.

**Storage:** a new `citizen_reports` table in the *same*
`data/history.sqlite3` database `history.py` already uses (not a new
storage mechanism). Only the fields listed in the module docstring are
kept — no unnecessary personal data is required or stored.

**"My Reports" has no login system** in this prototype: it is scoped to
the visitor's browser session (a list of Report IDs they've submitted).
Anyone with a Report ID can still look it up via *Track Report*, matching
how a real citizen-complaint number would work.

**Explicitly a prototype:** the Maintenance Dashboard does not connect to
any real municipal backend or authority, no training metrics/mAP/GPS/
municipal-data are fabricated anywhere in this module, and every report
still carries the same "decision-support, not certified inspection"
disclaimer as the rest of RoadScan.

## 23. Location-Aware Enhancement: GPS, Area Health, Damage Map, Duplicate Grouping, Segment History

Four additions layered on top of the existing pipeline and database — no
existing feature, route, or table was removed or replaced. Conceptually:

```
Image -> YOLO26 -> Damage Class + Confidence + Bounding Box
      -> Severity -> Road Health Score -> Repair Priority     (unchanged)
                              |
                              v
                          Location                <- geo.py
                              |
                              v
                        Road Segment               <- geo.make_segment_id
                              |
                              v
                       Area Aggregation             <- area_health.py
                              |
                              v
                       Area Scorecard  ---->  Damage Heatmap   <- damage_map.py
                                          \
                                           -> Citizen Reporting -> Duplicate
                                              Grouping (dedup.py, extended)
```

The Area Scorecard is an **aggregation layer over real stored observations,
not a new prediction model** — it never claims YOLO26 directly predicts an
area score.

### 23.1 GPS + Area Health Scorecard (PART 1)

- **Capture:** both *Inspect Road* (`/`) and *Report Road Damage*
  (`/report`) now show a shared **"Use My Location"** button
  (`templates/_location_fields.html`) that calls the browser Geolocation
  API only on click (never silently), plus always-available manual
  **Area / City / State / Road name / Landmark** fields. GPS is optional
  end-to-end; nothing requires it.
- **Reverse geocoding:** `geo.reverse_geocode()` calls the free OSM
  Nominatim API, server-side, with a 3s timeout; on any failure (offline,
  package missing, bad response) it returns `None` and the UI simply asks
  the person to fill Area/City/State manually. `/api/reverse-geocode` is
  the small JSON endpoint the location-fields script calls after a
  successful GPS capture.
- **Aggregation rule (never one photo for a whole area):**
  `area_health.get_area_scorecard()` computes the **Area Health Score as
  the mean of the latest Road Health Score for each distinct road segment**
  observed in that area, and always states *"Based on N inspected road
  segments"*. If there is no data for an area, the API/UI say
  `"Insufficient data"` — never a fabricated number.
- **Pages:** `/area-health` (list of known areas) and
  `/area-health/<area_key>` (full scorecard: score, condition,
  priority/resolved/pending counts, damage-type distribution,
  condition distribution, and a month-by-month trend — the trend is
  only shown with ≥2 distinct months of data).
- A compact **"Your Area"** card appears on the dashboard (`/`) once a
  session has submitted/inspected with an area (tracked via
  `session["last_area_key"]` — no account system, consistent with how
  "My Reports" already works).

### 23.2 Road Damage Heatmap (PART 2)

- `damage_map.py` collects every stored inspection/report that has a
  **real, non-null latitude & longitude** — nothing is plotted that
  wasn't actually stored with coordinates.
- `/damage-map` renders these as a **lightweight, dependency-free SVG
  lat/long scatter plot** (green/yellow/orange/red by repair priority),
  with a click-to-inspect info panel and a plain HTML table fallback.
  This is deliberately not a tiled real-world map (no new mapping
  dependency was added, per the "keep it lightweight" requirement) — but
  `damage_map.get_map_points()` already returns exactly the data shape a
  real tile-based map (e.g. Leaflet) would need, so swapping the
  rendering in `templates/damage_map.html` later doesn't touch the data
  layer.
- With zero geolocated records, the page shows *"No geolocated
  inspections available yet."* instead of an empty/broken map.

### 23.3 Duplicate Complaint Grouping (PART 3)

- **Extends `dedup.py`** (the existing per-image dHash matcher) rather
  than adding a second, unrelated system. The existing dHash matcher is
  kept as-is for its original job (repeat-photo de-duplication within one
  location's history); a new section of the same file adds
  citizen-report-to-citizen-report grouping, since that needs different
  evidence (two different citizens' photos of the same pothole look
  nothing alike pixel-wise).
- **Evidence used:** matching dominant damage class **and** either GPS
  proximity (`geo.haversine_m`, ≤40 m = *"likely related"*, ≤120 m =
  *"possible duplicate"*) or, when GPS is missing on either report,
  location-text similarity (`difflib`, stdlib only).
- **Grouping, never deletion:** `dedup.recompute_duplicate_groups()`
  clusters reports (union-find) and upserts stable `DG-00001`-style group
  ids into a new `duplicate_groups` table; reports keep their own
  `duplicate_group_id` but are never merged or deleted. It's re-run
  (cheap, O(n²) at prototype scale) after every new citizen report.
- **UI:** a *"Possible Duplicate Groups"* section on `/maintenance`, plus
  `/duplicate-groups/<group_id>` showing every member report with a link
  back to its own full report page.

### 23.4 Road Segment Health History (PART 4)

- **Segment identity (prototype, explicitly documented as such):**
  `geo.make_segment_id(road_name, area, city)` — slugified
  `road-name__area`, or a location-text fallback if no road name/area was
  given, or `None` if there's truly nothing to key on. This is a simple,
  deterministic stand-in for real GIS road segmentation, not a claim of
  verified segmentation.
- `segment.py` groups every Inspect-Road assessment and every citizen
  report sharing a `segment_id` into one ordered history.
- `/road/<segment_id>` shows the segment profile (current health/
  condition/priority, total inspections, first/last inspected, damage
  summary) and an SVG health-over-time chart, reusing the exact same
  inline-SVG technique `history_detail.html` already used for per-location
  trend charts. `/segments` lists every known segment.
- With fewer than 2 inspections for a segment, no trend/chart is
  fabricated — the page says so plainly.

### 23.5 Database changes

All changes are **additive**, applied by a new `migrations.py`
(`run_migrations()`, called once at `app.py` startup, idempotent, safe on
both a brand-new and an existing `data/history.sqlite3`):

- `assessments` (history.py) gains: `latitude, longitude, area, city,
  state, road_name, landmark, segment_id`
- `citizen_reports` (citizen_report.py) gains: `area, city, state,
  segment_id, duplicate_group_id`
- new table `duplicate_groups` (`group_id, damage_class, damage_label,
  location_label, confidence, member_count, created_at, updated_at`)

No existing column, table, or row is renamed, dropped, or overwritten;
pre-existing rows simply have `NULL` for the new columns until
re-inspected/re-reported with location data.

### 23.6 New modules / routes / pages summary

| New module | Responsibility |
|---|---|
| `geo.py` | distance, slugify, segment id, optional reverse geocoding |
| `migrations.py` | additive SQLite schema migration, run once at startup |
| `area_health.py` | Area Health Scorecard aggregation |
| `damage_map.py` | geolocated point collection for the Damage Map |
| `segment.py` | Road Segment Health History grouping |
| `dedup.py` (extended) | + citizen-report duplicate-complaint grouping |

| New route | Page |
|---|---|
| `GET /area-health` | list of known areas |
| `GET /area-health/<area_key>` | Area Health Scorecard |
| `GET /api/reverse-geocode` | JSON GPS→area/city/state (optional, best-effort) |
| `GET /damage-map` | Road Damage Heatmap |
| `GET /segments` | list of known road segments |
| `GET /road/<segment_id>` | Road Segment Profile |
| `GET /duplicate-groups/<group_id>` | duplicate-group detail |

`/analyze` and `/report/submit` (both pre-existing) now also accept
optional `latitude, longitude, area, city, state, road_name, landmark`
form fields and persist them.

## Future improvements

- Verify and benchmark the real YOLOv7 ensemble once weights are supplied.
- Swap in a trained segmentation head (e.g. YOLOv7-seg) for true pixel masks.
- Swap the evidence heatmap for real Grad-CAM/EigenCAM once YOLOv7 is active.
- Fit severity/health-score weights to human-labeled severity data if collected.
- Swap the Damage Map's SVG scatter plot for a real tiled map (e.g.
  Leaflet) — the data layer (`damage_map.py`) is already shaped for it.
- Real GIS road-segment boundaries instead of the current text-based
  `road_name + area` segment identity.

---

## MODULE-BY-MODULE EXPLANATION (viva reference)

**1. Dataset** — RDD2022 (D00/D10/D20/D40), used by the upstream trained
model; not bundled or modified here.

**2. Preprocessing** — resize to ≤1600px longest side (app.py); CLAHE
(heuristic) or letterbox (YOLOv7) at inference time.

**3. YOLOv7** — anchor-based single-stage CNN detector; here, an ensemble
of two resolutions (640/512) merged by NMS. *Input:* image tensor.
*Output:* boxes + class + confidence. *Limitation:* needs GPU-friendly
`torch`, and its checkpoints must be trained separately (not part of this repo).

**4. Detection** (`detector.py`) — turns raw model/heuristic output into
`Detection` objects with a shared interface, so every module downstream is
detector-agnostic. *Input:* BGR image (numpy array). *Output:* `List[Detection]`.

**5. Segmentation/visualization** (`explain.py`) — draws boxes + masks on
the image and a separate, honestly-labeled evidence heatmap. *Input:*
image + detections. *Output:* two annotated images.

**6. Severity** (`severity.py`) — per-detection 0–100 score from area,
confidence, and class weight → LOW/MODERATE/HIGH/CRITICAL + a reason
string. *Input:* detections + image shape. *Output:* `DamageAssessment` list.

**7. Health score** (`health_score.py`) — aggregates assessments into one
0–100 score with an additive, per-class-breakdown penalty model. *Input:*
assessments. *Output:* `HealthResult`.

**8. Repair priority** (`severity.compute_repair_priority`) — rule cascade
over assessments + health score → LOW/MEDIUM/HIGH/URGENT + reason +
recommended action.

**9. SQLite history** (`history.py`) — persists assessments per
user-declared location; computes improving/stable/deteriorating trend
between the two latest records only (never with <2 records).

**10. Before/after comparison** (`comparison.py`) — runs the pipeline on
two images, diffs the results, and gates the verdict on a scene-similarity
check so a false "repair succeeded" is never claimed from dissimilar photos.

**11. Report generation** (`report.py`) — formats the same data three ways
(JSON/HTML/PDF); no scoring logic lives here.

**12. Flask UI** (`app.py` + `templates/`) — orchestrates the above in
request handlers only; contains no detection/scoring logic itself.

**13. Citizen reporting** (`citizen_report.py`) — persists citizen-submitted
reports (new SQLite table, same DB file as `history.py`) and their status
lifecycle. *Input:* the same `run_pipeline()` output as any other analysis,
plus citizen-entered location/details. *Output:* a `CitizenReport` row with
a stable Report ID. Contains no detection/severity/health logic of its own.

**14. Location utilities** (`geo.py`) — GPS distance, segment/area
identity, optional reverse geocoding. Pure utility module; no detection/
scoring logic, no hard external-service dependency.

**15. Area Health Scorecard** (`area_health.py`) — aggregates stored
health scores per area (mean of latest-per-segment). *Input:* every
located `AssessmentRecord`/`CitizenReport`. *Output:* a scorecard dict, or
`None`/`insufficient_data` when there isn't enough real data.

**16. Damage Map** (`damage_map.py`) — filters stored records to those
with real coordinates and shapes them for the map page. No detection
logic; a pure data-selection layer.

**17. Road Segment History** (`segment.py`) — groups stored records by
`segment_id` into one ordered timeline. No detection/scoring logic.

**18. Duplicate grouping** (`dedup.py`, extended) — clusters citizen
reports via GPS/text/damage-class evidence into `duplicate_groups`; never
deletes or merges the underlying reports.

**19. Schema migration** (`migrations.py`) — additive `ALTER TABLE`/
`CREATE TABLE IF NOT EXISTS` runner, called once at `app.py` startup;
makes the database changes above safe for both fresh installs and
existing `data/history.sqlite3` files.

---

## Citizen-first redesign (current UX)

RoadScan is now a civic reporting platform: **report road damage in under a minute, track it, and earn recognition.**
The detection, severity, health-score, comparison, dedup and reporting engines are unchanged; the redesign is the
product layer on top of them.

| Area | URL | Notes |
|---|---|---|
| Home | `/` | Real counts only (reports, resolved, areas) |
| Report Damage | `/report` | Photo, then location (GPS or manual), then description, then submit |
| Track / My Reports | `/track`, `/reports`, `/reports/<id>` | Road-style progress tracker, responsible authority, repair evidence |
| Road Map | `/map` | India-wide Leaflet map, severity-coloured markers, filters, area side panel |
| Rewards / Profile | `/rewards`, `/profile` | Points, badges, prototype certificate (`/verify/<id>`) |
| Authority | `/authority/` | Dashboard, work queues, assignment, 14-day escalation, resolve with after-photo, analytics, technical tools |

**Prototype honesty.** Assignments, target dates and escalations are simulated inside RoadScan; nothing is sent to a
real government system. Certificates/badges are RoadScan recognition only. Citizen identity is a lightweight cookie
(`citizens.py`) so a real login can replace it later.

**Authority access.** By default the authority area is open (prototype mode, clearly bannered). Set
`ROADSCAN_AUTHORITY_KEY=<secret>` to require a key, and `ROADSCAN_SECRET_KEY=<random>` for signed sessions.

**Configuration** lives in `config.py` (SLA days, point values, badge thresholds, escalation ladder, severity colours).
Database changes are additive migrations (`migrations.py`) that run on start; existing data is preserved.

**Notes.** "Use My Location" needs HTTPS (or localhost) in modern browsers; on plain HTTP over a LAN it falls back to
manual entry. The map uses Leaflet + OpenStreetMap tiles from public CDNs (internet required). `pip install requests`
enables area-name lookup from GPS.

**Tests** (each runs on a temporary copy of the project and never touches your data):
`python3 tests/run_redesign_tests.py`, `tests/run_citizen_report_tests.py`, `tests/run_location_features_tests.py`.
