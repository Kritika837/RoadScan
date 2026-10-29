# RoadScan — AI-Powered Road Damage Detection & Civic Reporting

<p align="center">
  <strong>Detect damage. Assess road health. Prioritize repairs. Track reports.</strong>
</p>

<p align="center">
  <a href="https://roadscan-kccw.onrender.com/"><strong>🚀 Live Demo</strong></a> •
  <a href="https://github.com/Kritika837/RoadScan"><strong>💻 GitHub Repository</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-purple" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Framework-Flask-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Database-SQLite-blue" alt="SQLite">
  <img src="https://img.shields.io/badge/Deployment-Render-46E3B7?logo=render" alt="Render">
</p>

---

## 🌐 Live Application

**Try RoadScan:** https://roadscan-kccw.onrender.com/

RoadScan is a computer vision-powered road damage reporting and maintenance-prioritization prototype. It combines road image analysis with severity assessment, road health scoring, repair prioritization, geospatial visualization, report tracking, and a community rewards system.

The goal is to turn a road damage observation into a structured, traceable report rather than stopping at object detection alone.

## 🎯 Problem Statement

Road cracks, surface deterioration, and potholes can affect road safety, vehicle performance, and transportation infrastructure. Manual inspections and fragmented reporting can make it difficult to organize observations and track maintenance progress.

RoadScan explores a technology-assisted workflow that helps users:

- Identify and visualize potential road damage.
- Assess the estimated severity and condition of a road image.
- Assign maintenance priorities using transparent application rules.
- Submit and track structured damage reports.
- Visualize reported locations on a road map.
- Encourage community participation through points and rewards.

## ✨ Key Features

### 1. AI-Based Road Damage Analysis

- Road image analysis through a YOLOv7-based detection workflow.
- Fine-tuning work directed at the road damage detection task.
- Visualized detection regions and analysis results.
- Damage labels and confidence information where available in the detector output.

The supported road damage categories associated with the project's research foundation are:

| Label | Category | Description |
|---|---|---|
| D00 | Longitudinal crack | Cracks extending mainly along the road |
| D10 | Transverse crack | Cracks extending mainly across the road |
| D20 | Alligator crack | Interconnected cracks forming a network |
| D40 | Pothole | Localized road surface failure |

The categories above describe the research dataset's label scheme. The live application may display additional or differently named application-level categories.

### 2. Severity Assessment

RoadScan assigns a severity category to help users interpret the detected road condition.

The application can present severity levels such as Low, Moderate, High, and Critical, depending on its configured assessment rules.

Severity assessment is an application-level estimate and is not a certified engineering evaluation.

### 3. Road Health Score

- Produces a numerical road health score on a 0–100 scale.
- Summarizes the estimated condition of an analyzed road image.
- Supports quick interpretation of road condition alongside severity and repair priority.

**Example from the deployed application:** A submitted report displayed a road health score of **17.8/100**.

This is an example from one report, not an aggregate model-performance result.

### 4. Repair Priority

RoadScan converts assessment results into maintenance-priority labels to help users interpret which reported conditions may need attention first.

A deployed report displayed an **Urgent** repair priority. Priority is determined by the application's assessment logic and should support, not replace, professional inspection.

### 5. Civic Damage Reporting

- Submit a road damage report with a photograph and location information.
- Generate a unique report identifier.
- View a submission confirmation and report summary.
- Follow the report through a structured status workflow.

Example report identifier:

`RS-2026-000001`

The deployed prototype displays stages including Reported, Under Review, Assigned, In Progress, and Resolved.

**Transparency note:** RoadScan is a prototype civic-reporting workflow. It is not connected to a real government authority, and submitting a report through the application does not automatically dispatch a repair team.

### 6. Interactive Road Map

The Road Map interface provides:

- A map-based view of saved reports.
- Area search.
- Severity, status, damage type, and time filters.
- Summary statistics for reported road conditions.
- Links to nearby or listed reports.

This provides a geographic view of recorded observations rather than treating each report as an isolated event.

### 7. My Reports and Status Tracking

- Browse submitted reports.
- View report identifiers, locations, damage summaries, severity, and status.
- Open individual reports for more information.
- Follow the prototype's repair-status workflow.

### 8. Community Rewards

RoadScan includes a points-based participation system designed to encourage useful road damage reporting.

The deployed application displayed:

- **100 points** awarded for a valid submitted report.
- A progress indicator toward the next community badge.
- Recent reporting activity.
- Reward milestones and a certificate threshold.

Rewards are application features; they do not represent an official government incentive program.

### 9. Historical and Before-and-After Analysis

The broader RoadScan analysis workflow is designed to support historical observations and before-and-after image comparisons.

Comparisons should account for image similarity and capture conditions. A visual difference alone cannot establish that a repair succeeded.

### 10. Structured Reports and Explainability

The analysis pipeline is designed to organize detection results, severity estimates, road health scores, and repair priorities into interpretable outputs.

Depending on the enabled modules, report-generation support includes PDF, HTML, and JSON formats. Annotated images and assessment breakdowns help users review the basis of an application-generated result.

---

## 🔄 End-to-End Workflow

```text
Road Image / Damage Observation
              |
              v
       Image Processing
              |
              v
    Road Damage Detection
              |
              v
  Detection Results & Visualization
              |
              v
       Severity Assessment
              |
              v
        Road Health Score
              |
              v
      Repair Priority Rules
              |
              v
  Structured Damage Report
              |
       +------+------+
       |             |
       v             v
  Road Map      My Reports
       |             |
       +------+------+
              |
              v
      Status & Reward Workflow
```

The detection model identifies potential damage. Severity, health scoring, and repair prioritization are downstream application components rather than direct measures of model accuracy.

## 🧠 Model and Fine-Tuning

RoadScan's computer vision work is based on YOLOv7-style object detection and road damage recognition.

Fine-tuning adapts a pretrained model to a target task by continuing training on task-specific examples. For road damage detection, this involves learning to recognize damage categories and localize their positions in road images.

The model-development workflow includes:

1. Preparing road images and object-detection annotations.
2. Mapping annotation labels to the supported damage categories.
3. Fine-tuning the detector for the road damage task.
4. Evaluating detection outputs on road images.
5. Integrating detections with the RoadScan assessment and reporting workflow.

### Training and evaluation transparency

The exact training split, hyperparameters, final checkpoint filename, and numerical evaluation results are not documented in this README because those records have not yet been recovered.

No precision, recall, or mAP figures are claimed here. Application screenshots demonstrate interface workflows, not model accuracy.

Once the training artifacts are available, this section should document:

| Item | Details to record |
|---|---|
| Dataset | Exact dataset and version used |
| Data split | Training, validation, and test image counts |
| Starting weights | Pretrained checkpoint used for fine-tuning |
| Training configuration | Image size, batch size, epochs, optimizer, and learning rate |
| Final checkpoint | Actual saved weights filename |
| Evaluation | Precision, recall, mAP@0.5, and mAP@0.5:0.95 |
| Inference | Hardware, image size, and measured inference latency |

This makes the distinction between the model-development process and the deployed application clear and reproducible.

## 🏗️ System Architecture

RoadScan separates its responsibilities into distinct components.

| Component | Responsibility |
|---|---|
| Flask application | Routes, request handling, and application orchestration |
| Detection module | Identifies potential road damage |
| Image visualization | Displays annotated images and detection regions |
| Severity assessment | Estimates damage severity |
| Road health scoring | Produces a numerical condition indicator |
| Repair priority | Applies application-level maintenance rules |
| Reporting workflow | Creates and tracks damage reports |
| Road map | Visualizes saved reports geographically |
| Database layer | Persists supported application records |
| Rewards system | Tracks participation points and milestones |
| Reporting module | Organizes analysis into structured outputs |

## 🛠️ Technology Stack

- **Language:** Python
- **Web framework:** Flask
- **Computer vision:** OpenCV
- **Deep learning:** PyTorch and YOLOv7-based detection
- **Database:** SQLite
- **Reporting:** HTML, JSON, and PDF-generation tooling
- **Deployment:** Render

Refer to `requirements.txt` and the source code for the exact installed dependencies and versions.

## 🚀 Run Locally

### Prerequisites

- Python version compatible with the project's dependencies.
- Git.
- Sufficient memory and disk space for the dependencies and model weights.
- The model checkpoint and configuration required by the configured detector.

### 1. Clone the repository

```bash
git clone https://github.com/Kritika837/RoadScan.git
cd RoadScan
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the detector

Ensure the required model weights are available at the configured path. Verify any required environment variables and model dependencies before running inference.

Cloning the repository does not necessarily download the model checkpoint, particularly if large weight files are excluded from version control.

### 5. Start the Flask application

```bash
python app.py
```

Open the local address displayed by the application.

If the entry-point filename or startup procedure changes, follow the current repository configuration.

## 🧪 Application Validation

The deployed application has been exercised through visible reporting workflows, including:

- Submitting a road damage report.
- Displaying the submission confirmation and unique report ID.
- Viewing saved reports in My Reports.
- Displaying report information on the Road Map.
- Showing severity, road health, and repair priority.
- Awarding points for a valid report.
- Displaying the report-status workflow.

These are application-level observations. They should not be interpreted as a formal model benchmark or evidence that reports are transmitted to an external authority.

### Recommended model evaluation

For a reproducible computer vision evaluation, measure:

- **Precision:** The proportion of predicted damage detections that are correct.
- **Recall:** The proportion of labeled damage instances that are detected.
- **mAP@0.5:** Mean average precision at an intersection-over-union threshold of 0.5.
- **mAP@0.5:0.95:** Mean average precision averaged across multiple IoU thresholds.
- **Inference latency:** Time required to process an image under a documented setup.

Evaluation should use a held-out test set and clearly state the dataset, split, model checkpoint, and inference configuration.

## 📚 Research References

### Road damage dataset

Arya, D., et al. (2022). *RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection.*

- Paper: https://arxiv.org/abs/2209.08538
- Dataset information: https://figshare.com/

### Object detection model

Wang, C.-Y., Bochkovskiy, A., and Liao, H.-Y. M. *YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors.*

- Paper: https://arxiv.org/abs/2207.02696

These works provide research context for road damage detection and object detection. The exact dataset configuration and training results for RoadScan should be documented from the project's own artifacts.

## ⚠️ Limitations

- Detection quality depends on image quality, lighting, viewpoint, road conditions, and model generalization.
- Supported labels do not cover every possible road defect.
- Severity and health scores are application-defined estimates rather than certified engineering measurements.
- Repair-priority recommendations should be reviewed by qualified personnel.
- Historical comparisons require sufficiently comparable images and reliable location information.
- The civic-reporting workflow is a prototype and is not integrated with an official government repair system.
- Live deployment does not, by itself, establish model accuracy or real-world repair effectiveness.

## 🔮 Future Improvements

- Publish reproducible fine-tuning configurations and model evaluation results.
- Evaluate the detector on an independently labeled test set.
- Add GPS coordinates and timestamped road observations.
- Improve longitudinal tracking of road conditions.
- Validate severity and priority rules against expert assessments.
- Measure inference latency and deployment resource usage.
- Improve report verification and moderation workflows.
- Explore integration with authorized civic-maintenance systems.

## 👩‍💻 Project Details

**Project:** RoadScan — AI-Powered Road Damage Detection & Civic Reporting

**Domain:** Computer Vision, Deep Learning, and Civic Technology

**Developer:** Kritika Singh

**GitHub:** https://github.com/Kritika837/RoadScan

**Live application:** https://roadscan-kccw.onrender.com/

---

<p align="center">
  <strong>RoadScan — From road damage detection to actionable reporting.</strong>
</p>
