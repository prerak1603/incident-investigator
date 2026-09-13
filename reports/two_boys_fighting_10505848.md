# Incident Report — two_boys_fighting_10505848

Camera: cam_01  
Source: `data/raw/Fighting/two_boys_fighting_10505848.mp4`  
Duration: 12.56s

1 incident(s) detected: 0 confirmed, 0 downgraded, 1 needs human review. Idle time: 0.00s (0.0% of footage).

## Fighting — cam_01 @ 1.0s-12.0s

| raw | revised | final | verdict |
|---|---|---|---|
| 0.91 | 0.45 | 0.45 | needs_human_review |

peak person count 3, avg motion magnitude 16.53; historical false-positive rate at this camera/hour: 0.50

> The first-pass reasoning is internally consistent and the revised confidence of 0.45 is well-supported: high raw confidence is appropriately discounted by a 0.50 historical false positive rate, borderline person count, and absence of corroborating signals. The downgrade logic holds. However, the resulting confidence sits squarely in the ambiguous middle range (neither clearly below a dismiss threshold nor above a confirm threshold), the scenario involves potential violence, and there is only a single segment with no cross-validation possible within the case. These factors — violence class, unresolvable ambiguity from automated signals alone, and no corroboration path — collectively meet the bar for escalating to human review rather than auto-resolving as insufficient evidence. No inconsistency exists between segments since there is only one. Final confidence is held at 0.45 per the first-pass calculation, which remains sound.
