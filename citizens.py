"""
citizens.py
-----------
MODULE: Citizen identity (prototype, login-ready)
WHY:    The app has no authentication. Instead of pretending accounts exist,
        every part of the citizen UI (My Reports, Rewards, Profile) asks ONE
        function - `current_citizen()` - "who is this?". Today the answer is
        a lightweight local identity remembered by the visitor's cookie. When
        a real login system is added, only this function has to change; the
        reports, reward ledger and certificates already point at
        `citizens.id`.
LIMITATIONS: a cleared cookie starts a new identity (the data stays in the
        database, unowned). Nothing here is verified or secure - it is a
        convenience identity, not authentication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import citizen_report as creport

SESSION_KEY = "citizen_id"


@dataclass
class Citizen:
    id: int
    name: Optional[str]
    created_at: float

    @property
    def display_name(self) -> str:
        return self.name or "Citizen"

    @property
    def initial(self) -> str:
        return (self.name or "C").strip()[:1].upper()

    @property
    def code(self) -> str:
        return f"C-{self.id:05d}"


def _row(r) -> Citizen:
    return Citizen(id=r["id"], name=r["name"], created_at=r["created_at"])


def get(citizen_id: Optional[int]) -> Optional[Citizen]:
    if not citizen_id:
        return None
    conn = creport.get_conn()
    r = conn.execute("SELECT * FROM citizens WHERE id = ?", (citizen_id,)).fetchone()
    conn.close()
    return _row(r) if r else None


def create(name: Optional[str] = None) -> Citizen:
    now = time.time()
    conn = creport.get_conn()
    cur = conn.execute("INSERT INTO citizens (name, is_demo, created_at, updated_at) VALUES (?, 1, ?, ?)",
                       ((name or "").strip() or None, now, now))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return get(cid)


def set_name(citizen_id: int, name: str) -> None:
    conn = creport.get_conn()
    conn.execute("UPDATE citizens SET name = ?, updated_at = ? WHERE id = ?",
                 ((name or "").strip()[:60] or None, time.time(), citizen_id))
    conn.commit()
    conn.close()


def current_citizen(session, create_if_missing: bool = False) -> Optional[Citizen]:
    """
    The single 'who is this?' seam. Reading pages never creates identities
    (so casual visitors/bots don't fill the table); submitting a report or
    editing the profile does. A leftover list of report IDs from the older
    session-based 'My Reports' is adopted once, so nothing is lost.
    """
    c = get(session.get(SESSION_KEY))
    if c is None and create_if_missing:
        c = create()
        session[SESSION_KEY] = c.id
        session.permanent = True
    if c is not None:
        legacy = session.pop("my_reports", None)
        if legacy:
            creport.adopt_reports(list(legacy), c.id)
    return c
