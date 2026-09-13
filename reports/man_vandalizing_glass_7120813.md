# Incident Report — man_vandalizing_glass_7120813

Camera: cam_01  
Source: `data/raw/Vandalism/man_vandalizing_glass_7120813.mp4`  
Duration: 34.56s

1 incident(s) detected: 0 confirmed, 0 downgraded, 1 needs human review. Idle time: 9.56s (27.7% of footage).

## Vandalism — cam_01 @ 4.0s-29.0s

| raw | revised | final | verdict |
|---|---|---|---|
| 0.86 | 0.52 | 0.52 | needs_human_review |

peak person count 4, avg motion magnitude 6.48; historical false-positive rate at this camera/hour: no history

> The first-pass reasoning is internally consistent: it correctly identifies that motion magnitude, person count, and duration are all ambiguous signals that fit vandalism but equally fit benign activity, and it appropriately penalizes the absence of calibration data and cross-reference cases. The revised confidence of 0.52 follows logically from that reasoning — sitting near the uncertain midpoint — so there is no mismatch between reasoning and score. However, a raw model confidence of 0.86 dropping to 0.52 on a vandalism classification, with no contradicting evidence (only absent evidence), represents meaningful residual risk. Vandalism has a higher cost-of-miss than many classes, and the first-pass model explicitly flags insufficient corroboration rather than active contradiction. Auto-resolving a potential vandalism event with a coin-flip confidence and no calibration baseline is inadvisable. The case should be escalated to a human reviewer who can inspect the actual footage for object interactions, surface contact, or other scene-specific cues that the numeric features alone cannot capture. No inter-segment inconsistencies exist since this is a single-segment case.
