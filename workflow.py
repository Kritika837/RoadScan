"""
workflow.py
-----------
MODULE: Report lifecycle rules (verification, assignment, escalation, resolution)
WHY:    routes should not contain business rules. Everything that changes a
        report after submission goes through here, so the audit trail
        (who / when / note) and the reward hooks are applied consistently.
        This module never runs detection or scoring - it only records what
        people (or the clock) did to a report.
PROTOTYPE HONESTY: assignment and escalation are SIMULATED. Nothing here
        contacts a real government system; the "Assigned" department and
        officer are simply what an authority user typed in, and
        "escalation" only records that the next level in
        config.ESCALATION_LEVELS should be involved. Until an integration
        exists, the UI says so.
Every function returns (ok: bool, message: str).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import citizen_report as creport
import config
import rewards

SYSTEM = "RoadScan system"
Result = Tuple[bool, str]


# ------------------------------------------------------------ submission --

def submission_block_reason(rep) -> Optional[str]:
    """Why this report should NOT earn the 'valid report' points (None = it may)."""
    if not rep.citizen_id:
        return "no_owner"
    earlier = [r for r in creport.find_by_hash(rep.image_hash) if r.id != rep.id and r.id < rep.id]
    if rep.image_hash and earlier:
        return "repeat_photo"
    if rep.duplicate_group_id:
        if any(m.citizen_id == rep.citizen_id and m.id < rep.id
               for m in creport.list_by_group(rep.duplicate_group_id)):
            return "own_duplicate"
    if rewards.rewarded_today(rep.citizen_id) >= config.DAILY_REWARDED_REPORT_CAP:
        return "daily_cap"
    return None


def award_submission(report_id: str) -> Tuple[int, Optional[str]]:
    """
    Called right after a report is stored (and duplicate groups recomputed).
    Returns (points_awarded, reason_if_none). A report with no detected
    damage is accepted but earns nothing until an authority verifies it.
    """
    rep = creport.get_report(report_id)
    if not rep:
        return 0, "missing"
    block = submission_block_reason(rep)
    if block:
        return 0, block
    if not rep.detections or rep.severity_level in (None, "NONE"):
        return 0, "no_damage"
    return rewards.award(rep.citizen_id, report_id, "REPORT_SUBMITTED"), None


# ------------------------------------------------------------ verification --

def _mark_verified(rep, by: str) -> int:
    """Record verification once and pay out; also pays the submission reward
    retroactively when a human confirmed a report the AI had not flagged."""
    if rep.is_verified:
        return 0
    creport.update_report(rep.report_id, {"verified_at": time.time(), "verified_by": by},
                          {"event": "Report verified", "by": by})
    pts = rewards.award(rep.citizen_id, rep.report_id, "REPORT_VERIFIED")
    if not submission_block_reason(rep):
        pts += rewards.award(rep.citizen_id, rep.report_id, "REPORT_SUBMITTED")
    return pts


def verify(report_id: str, by: str, note: Optional[str] = None) -> Result:
    rep = creport.get_report(report_id)
    if not rep:
        return False, "Report not found."
    if rep.status == "RESOLVED":
        return False, "This report is already resolved."
    _mark_verified(rep, by)
    if rep.status == "REPORTED":
        creport.update_status(report_id, "UNDER_REVIEW", by=by, note=note or "Report verified and under review")
    return True, f"{report_id} verified and moved to Under Review."


# -------------------------------------------------------------- assignment --

def assign(report_id: str, department: str, officer_name: str, officer_id: str,
           by: str, note: Optional[str] = None) -> Result:
    rep = creport.get_report(report_id)
    if not rep:
        return False, "Report not found."
    if rep.status == "RESOLVED":
        return False, "This report is already resolved."
    department, officer_name, officer_id = (department or "").strip(), (officer_name or "").strip(), (officer_id or "").strip()
    if not department:
        return False, "Enter the responsible department."
    _mark_verified(rep, by)
    summary = department + (f" - {officer_name}" if officer_name else "")
    fields = {"department": department, "officer_name": officer_name or None,
              "officer_id": officer_id or None, "assigned_at": time.time()}
    if rep.status in ("REPORTED", "UNDER_REVIEW"):
        fields["status"] = "ASSIGNED"
        creport.update_report(report_id, fields,
                              {"status": "ASSIGNED", "by": by, "note": note or f"Assigned to {summary}"})
    else:
        creport.update_report(report_id, fields,
                              {"event": f"Reassigned to {summary}", "by": by, "note": note})
    return True, f"{report_id} assigned to {summary}."


# ------------------------------------------------------------------ status --

def set_status(report_id: str, new_status: str, by: str, note: Optional[str] = None) -> Result:
    rep = creport.get_report(report_id)
    if not rep:
        return False, "Report not found."
    if new_status == "RESOLVED":
        return False, "Use 'Resolve' - repair evidence (an after photo) is required."
    if new_status not in creport.STATUSES:
        return False, "Unknown status."
    if new_status in ("ASSIGNED", "IN_PROGRESS") and not rep.is_assigned:
        return False, "Assign a responsible department first."
    if new_status == rep.status:
        return False, f"Already {creport.STATUS_LABELS[new_status]}."
    if new_status != "REPORTED":
        _mark_verified(rep, by)
    creport.update_status(report_id, new_status, by=by, note=note)
    return True, f"{report_id} is now {creport.STATUS_LABELS[new_status]}."


# -------------------------------------------------------------- escalation --

def sync_overdue() -> int:
    """Flag every open report that has passed its target (once each, audited)."""
    n = 0
    now = time.time()
    for r in creport.list_all():
        if r.is_overdue(now) and not r.overdue_flagged_at:
            creport.update_report(
                r.report_id, {"overdue_flagged_at": now},
                {"event": "Resolution target exceeded - escalation required", "by": SYSTEM})
            n += 1
    return n


def escalate(report_id: str, by: str, note: Optional[str] = None) -> Result:
    rep = creport.get_report(report_id)
    if not rep:
        return False, "Report not found."
    if not rep.is_overdue():
        return False, "Only reports past their resolution target can be escalated."
    nxt = rep.next_escalation_name
    if not nxt:
        return False, "Already at the highest escalation level."
    creport.update_report(
        report_id, {"escalation_level": (rep.escalation_level or 0) + 1, "escalated_at": time.time()},
        {"event": f"Escalated to {nxt} (simulated - no external authority was notified)",
         "by": by, "note": note})
    return True, f"{report_id} escalated to {nxt} (simulated)."


# -------------------------------------------------------------- resolution --

def resolve(report_id: str, by: str, after_image_path: str, after_annotated_path: Optional[str],
            comparison: Optional[dict], notes: Optional[str], resolution_date: Optional[float]) -> Result:
    rep = creport.get_report(report_id)
    if not rep:
        return False, "Report not found."
    if rep.status == "RESOLVED":
        return False, "This report is already resolved."
    if not after_image_path:
        return False, "Repair evidence (an after photo) is required to resolve a report."
    if resolution_date and resolution_date < rep.created_at - 86400:
        return False, "The resolution date cannot be before the report was submitted."
    if resolution_date and resolution_date > time.time() + 86400:
        return False, "The resolution date cannot be in the future."
    _mark_verified(rep, by)
    creport.resolve_report(report_id, after_image_path, after_annotated_path, comparison,
                           by=by, notes=notes, resolution_date=resolution_date)
    pts = rewards.award(rep.citizen_id, report_id, "REPORT_RESOLVED")
    msg = f"{report_id} resolved with repair evidence."
    if pts:
        msg += f" The reporting citizen earned +{pts} points."
    return True, msg


def confirm_repair(report_id: str, citizen_id: int, photo_path: str) -> Result:
    """Citizen follow-up: a photo after the repair. Rewarded once per report."""
    rep = creport.get_report(report_id)
    if not rep or rep.citizen_id != citizen_id:
        return False, "Only the person who reported this can confirm the repair."
    if rep.status != "RESOLVED":
        return False, "The repair can be confirmed once the report is resolved."
    if rep.citizen_followup_path:
        return False, "You have already added a follow-up photo."
    creport.update_report(report_id, {"citizen_followup_path": photo_path, "followup_at": time.time()},
                          {"event": "Citizen added a follow-up photo after the repair", "by": "Citizen"})
    pts = rewards.award(citizen_id, report_id, "REPAIR_FOLLOWUP")
    return True, f"Thanks for confirming the repair!" + (f" +{pts} points." if pts else "")


# ---------------------------------------------------------------- display --

def build_timeline(rep) -> list:
    """Steps for the visual progress tracker: done / current / todo, with the time each was reached."""
    times = rep.step_times()
    idx = rep.status_step_index
    steps = []
    for i, key in enumerate(creport.STATUSES):
        if rep.status == "RESOLVED":
            state = "done"
        else:
            state = "done" if i < idx else ("current" if i == idx else "todo")
        steps.append({"key": key, "label": creport.STATUS_LABELS[key], "state": state, "time": times.get(key)})
    return steps
