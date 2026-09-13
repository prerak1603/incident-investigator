import json
import os
from dataclasses import dataclass, field

import anthropic

from incident_investigator.cases import Case
from incident_investigator.memory import NO_MEMORY, SHORT_TERM_ONLY, FULL_MEMORY

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1000

SYSTEM_PROMPT = (
    "You are reviewing candidate incident detections flagged by a CCTV analysis "
    "pipeline (CLIP zero-shot classification, YOLO person counting, optical flow "
    "motion). Your job is to re-evaluate the raw detections against the evidence "
    "given and decide whether each segment actually corroborates the claimed "
    "activity, contradicts it, or the evidence is insufficient either way. "
    "Output ONLY valid JSON, no markdown fences, no preamble, no commentary."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are the second-pass reviewer in an incident-detection pipeline. A first-pass "
    "model already produced a verdict for this case. Your job is to audit that verdict, "
    "not re-derive it from scratch: check whether its stated reasoning actually supports "
    "the status and confidence it assigned, whether segments in the same case contradict "
    "each other, and whether the case should be escalated to a human instead of "
    "auto-resolved. If the first-pass verdict holds up, say so — do not change it just to "
    "produce a change. Output ONLY valid JSON, no markdown fences, no preamble, no commentary."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


@dataclass
class SegmentVerdict:
    segment_class: str
    revised_confidence: float
    corroboration_status: str
    reasoning: str


@dataclass
class CaseVerdict:
    case: Case
    segments: list = field(default_factory=list)
    needs_review: bool = False
    raw_response: str = ""
    error: str = None


def _build_prompt(case, memory_mode=FULL_MEMORY):
    lines = [f"Case spanning {case.start:.1f}s-{case.end:.1f}s on camera {case.camera_id}:\n"]

    for seg in case.segments:
        line = (
            f"- segment class={seg.label} raw_confidence={seg.confidence:.2f} "
            f"duration={seg.end - seg.start:.1f}s peak_person_count={seg.peak_person_count} "
            f"avg_motion_magnitude={seg.avg_motion_magnitude:.2f}"
        )
        if memory_mode == FULL_MEMORY:
            fp_rate = case.historical_fp_rate.get(seg.label)
            fp_str = "no history for this camera/class/hour" if fp_rate is None else f"{fp_rate:.2f}"
            line += f" historical_false_positive_rate={fp_str}"
        lines.append(line)
        if seg.label == "fighting" and seg.peak_person_count < 2:
            lines.append(
                f"  note: classified as fighting but only {seg.peak_person_count} person(s) "
                "were ever detected in frame during this segment"
            )

    if memory_mode in (SHORT_TERM_ONLY, FULL_MEMORY):
        if case.related_cases:
            lines.append(f"\n{len(case.related_cases)} related case(s) recently seen on this camera:")
            for rc in case.related_cases:
                lines.append(f"- classes={rc.classes} at {rc.start:.1f}s-{rc.end:.1f}s")
        else:
            lines.append("\nNo related recent cases in short-term memory.")

    lines.append(
        "\nFor each segment return revised_confidence (float 0-1), corroboration_status "
        '(one of "corroborated", "contradicted", "insufficient_evidence"), and a short '
        "reasoning string. Respond with exactly this JSON shape:\n"
        '{"segments": [{"class": "...", "revised_confidence": 0.0, '
        '"corroboration_status": "...", "reasoning": "..."}]}'
    )
    return "\n".join(lines)


def evaluate_case(case: Case, memory_mode: str = FULL_MEMORY) -> CaseVerdict:
    prompt = _build_prompt(case, memory_mode)

    try:
        client = _get_client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
    except Exception as e:
        return CaseVerdict(case=case, needs_review=True, error=f"API call failed: {e}")

    try:
        parsed = json.loads(raw)
        segments = [
            SegmentVerdict(
                segment_class=s["class"],
                revised_confidence=float(s["revised_confidence"]),
                corroboration_status=s["corroboration_status"],
                reasoning=s["reasoning"],
            )
            for s in parsed["segments"]
        ]
    except Exception as e:
        return CaseVerdict(case=case, needs_review=True, raw_response=raw, error=f"failed to parse response: {e}")

    return CaseVerdict(case=case, segments=segments, raw_response=raw)


@dataclass
class ReflectedSegment:
    segment_class: str
    pre_confidence: float
    pre_status: str
    final_confidence: float
    final_verdict: str
    reflection_note: str


@dataclass
class CaseReflection:
    case: Case
    segments: list = field(default_factory=list)
    needs_review: bool = False
    raw_response: str = ""
    error: str = None


def _build_reflection_prompt(case, verdict):
    lines = [
        f"Case spanning {case.start:.1f}s-{case.end:.1f}s on camera {case.camera_id}. "
        "A first-pass model already reviewed this case and produced the verdicts below. "
        "Audit its work, don't just restate it."
    ]

    for seg, sv in zip(case.segments, verdict.segments):
        lines.append(
            f"\n- segment class={seg.label} raw_confidence={seg.confidence:.2f} "
            f"duration={seg.end - seg.start:.1f}s peak_person_count={seg.peak_person_count} "
            f"avg_motion_magnitude={seg.avg_motion_magnitude:.2f}"
        )
        lines.append(
            f"  first-pass verdict: revised_confidence={sv.revised_confidence:.2f} "
            f"corroboration_status={sv.corroboration_status}"
        )
        lines.append(f"  first-pass reasoning: {sv.reasoning}")

    lines.append(
        "\nFor each segment check: (1) does the stated reasoning actually support the "
        "corroboration_status and confidence assigned, or is there a mismatch — e.g. the "
        "reasoning argues the evidence is weak but confidence was left high? (2) is there "
        "any inconsistency between segments in this same case? (3) should this be escalated "
        "to explicit human review rather than auto-resolved?\n"
        "For each segment return final_confidence (float 0-1), final_verdict (one of "
        '"confirmed", "downgraded", "needs_human_review"), and reflection_note explaining '
        "what changed, if anything, and why. Respond with exactly this JSON shape:\n"
        '{"segments": [{"class": "...", "final_confidence": 0.0, "final_verdict": "...", '
        '"reflection_note": "..."}]}'
    )
    return "\n".join(lines)


def reflect_case(case: Case, verdict: CaseVerdict) -> CaseReflection:
    if verdict.needs_review:
        return CaseReflection(case=case, needs_review=True, error="no stage 5 verdict to reflect on")

    prompt = _build_reflection_prompt(case, verdict)

    try:
        client = _get_client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
    except Exception as e:
        return CaseReflection(case=case, needs_review=True, error=f"API call failed: {e}")

    try:
        parsed = json.loads(raw)
        pre = {sv.segment_class: sv for sv in verdict.segments}
        segments = [
            ReflectedSegment(
                segment_class=s["class"],
                pre_confidence=pre[s["class"]].revised_confidence,
                pre_status=pre[s["class"]].corroboration_status,
                final_confidence=float(s["final_confidence"]),
                final_verdict=s["final_verdict"],
                reflection_note=s["reflection_note"],
            )
            for s in parsed["segments"]
        ]
    except Exception as e:
        return CaseReflection(case=case, needs_review=True, raw_response=raw, error=f"failed to parse response: {e}")

    return CaseReflection(case=case, segments=segments, raw_response=raw)
