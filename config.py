"""
config.py
---------
Single place for the product-level constants of the citizen workflow:
reward points, badges, the resolution target (SLA), escalation levels and
map colours. Nothing here touches detection, severity or Road Health
scoring - those stay in detector.py / severity.py / health_score.py.

Change a number here and it changes everywhere (UI copy included), so no
templates or routes hard-code points or day counts.
"""

# --- Resolution target (prototype workflow, not a legal SLA) ---------------
SLA_DAYS = 14

# --- Reward points (awarded once per report per event type) ----------------
POINTS = {
    "REPORT_SUBMITTED": 100,   # valid report: damage detected + unique photo
    "REPORT_VERIFIED": 200,    # an authority reviewed / verified the report
    "REPORT_RESOLVED": 500,    # the report ended in a documented repair
    "REPAIR_FOLLOWUP": 100,    # citizen confirms the repair with a photo
}
POINT_LABELS = {
    "REPORT_SUBMITTED": "Valid report submitted",
    "REPORT_VERIFIED": "Report verified",
    "REPORT_RESOLVED": "Issue resolved",
    "REPAIR_FOLLOWUP": "Repair follow-up submitted",
}
POINT_HELP = {
    "REPORT_SUBMITTED": "Send a clear photo where road damage is visible. Repeated or identical photos earn nothing extra.",
    "REPORT_VERIFIED": "Awarded when your report has been reviewed and verified.",
    "REPORT_RESOLVED": "Awarded when the road is repaired and repair evidence is recorded.",
    "REPAIR_FOLLOWUP": "Add a photo after the repair to confirm it on the ground.",
}
# Anti-abuse: at most this many reports per citizen per day can earn the
# "valid report" points (the reports themselves are never blocked).
DAILY_REWARDED_REPORT_CAP = 10
# Same citizen + identical photo inside this window = treated as a double tap.
DOUBLE_SUBMIT_WINDOW_SECONDS = 120

# --- Citizen levels / badges (ordered by min_points; add more freely) ------
BADGES = [
    {"key": "community_member", "name": "Community Member", "min_points": 0, "icon": "\U0001F91D",
     "tagline": "Welcome aboard - every report helps."},
    {"key": "road_watcher", "name": "Road Watcher", "min_points": 1000, "icon": "\U0001F440",
     "tagline": "Keeping an eye on the roads around you."},
    {"key": "civic_champion", "name": "Civic Champion", "min_points": 5000, "icon": "\U0001F31F",
     "tagline": "A dependable voice for safer streets."},
    {"key": "road_saver", "name": "Road Saver", "min_points": 10000, "icon": "\U0001F3C6",
     "tagline": "Recognized for actively contributing to safer roads."},
]
CERTIFICATE_BADGE_KEY = "road_saver"      # badge level that unlocks the certificate
CERTIFICATE_TITLE = "RoadScan Civic Contribution Certificate"
CERTIFICATE_NOTE = ("Prototype recognition issued by RoadScan. This is not a government-issued "
                    "certificate and carries no official authority.")

# --- Escalation ladder (simulated until a real authority backend exists) ---
ESCALATION_LEVELS = [
    "Local Authority",
    "Municipal / Block Authority",
    "District Authority",
    "Higher Authority",
]

# Suggestions only - the authority user may type any department name.
DEPARTMENT_SUGGESTIONS = [
    "Municipal Roads Department",
    "Public Works Department",
    "District Roads Authority",
    "Highway Authority",
]

# --- Map colours (severity) --------------------------------------------------
SEVERITY_COLORS = {
    "LOW": "#2E9E5B",
    "MODERATE": "#F2C200",
    "HIGH": "#F07C1A",
    "CRITICAL": "#D32F2F",
    "NONE": "#8A97A3",
}
SEVERITY_ORDER = ["NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"]

DAMAGE_TYPES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
TIME_FILTERS = {"7": 7, "30": 30, "180": 183}   # map "Time" filter -> days
