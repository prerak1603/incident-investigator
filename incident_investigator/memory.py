import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

DB_PATH = "data/long_term_memory.db"

NO_MEMORY = "none"
SHORT_TERM_ONLY = "short_term"
FULL_MEMORY = "short_and_long_term"
MEMORY_MODES = (NO_MEMORY, SHORT_TERM_ONLY, FULL_MEMORY)


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            class TEXT,
            hour_of_day INTEGER,
            verdict TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def seed_demo_data(conn):
    # placeholder rows — no real incident history exists yet, this only exists
    # so false_positive_rate() has something to look up before Stage 5/6 start
    # writing real verdicts back into this table
    if conn.execute("SELECT COUNT(*) FROM case_history").fetchone()[0] > 0:
        return
    rows = [
        ("cam_01", "fighting", 22, "false_positive", 0.61, "2026-09-01T22:14:00"),
        ("cam_01", "fighting", 22, "true_positive", 0.89, "2026-09-03T22:47:00"),
        ("cam_01", "fire_smoke", 22, "false_positive", 0.44, "2026-09-06T22:02:00"),
        ("cam_01", "vandalism", 3, "false_positive", 0.55, "2026-09-05T03:20:00"),
    ]
    conn.executemany(
        "INSERT INTO case_history (camera_id, class, hour_of_day, verdict, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def false_positive_rate(conn, camera_id, class_, hour_of_day):
    cur = conn.execute(
        "SELECT verdict FROM case_history WHERE camera_id = ? AND class = ? AND hour_of_day = ?",
        (camera_id, class_, hour_of_day),
    )
    verdicts = [row[0] for row in cur.fetchall()]
    if not verdicts:
        return None
    return verdicts.count("false_positive") / len(verdicts)


@dataclass
class ShortTermMemory:
    max_cases: int = 50
    cases: list = field(default_factory=list)

    def add(self, case):
        self.cases.append(case)
        if len(self.cases) > self.max_cases:
            self.cases.pop(0)

    def related(self, case, window_seconds=60.0):
        return [c for c in self.cases if c is not case and abs(c.start - case.start) <= window_seconds]


def attach_memory(cases, conn, short_term, reference_time=None, memory_mode=FULL_MEMORY):
    reference_time = reference_time or datetime.now()
    for case in cases:
        case_hour = (reference_time + timedelta(seconds=case.start)).hour

        if memory_mode == FULL_MEMORY:
            for label in case.classes:
                case.historical_fp_rate[label] = false_positive_rate(conn, case.camera_id, label, case_hour)

        if memory_mode in (SHORT_TERM_ONLY, FULL_MEMORY):
            case.related_cases = short_term.related(case)

        # bookkeeping happens regardless of mode — a "none" run just never reads
        # this back out, it isn't pretending the memory store doesn't exist
        short_term.add(case)
    return cases
