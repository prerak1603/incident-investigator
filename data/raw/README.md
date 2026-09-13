# Sample clips — placeholder dataset

**These are NOT UCF-Crime dataset clips.** Kaggle auth could not be established
(invalid/placeholder API tokens across three attempts), so per request these were
substituted with free stock footage from [Pexels](https://www.pexels.com) (CDN:
`videos.pexels.com`) to unblock pipeline testing. Swap this folder for the real
UCF-Crime subset before drawing any conclusions about detection accuracy.

Category folders map loosely to UCF-Crime anomaly classes but the content is
generic stand-in footage, not real crime/incident recordings:

- `Fighting/` — staged sparring / boxing, not real physical altercations.
- `Vandalism/` — graffiti / spray-paint / property damage.
- `Arson/` — general building fire and smoke footage, not deliberate-ignition incidents.
- `Normal/` — ordinary street/pedestrian footage, no incident.

## Fighting
| file | source |
|---|---|
| `two_boys_fighting_10505848.mp4` | https://www.pexels.com/video/two-boys-fighting-on-the-ground-10505848/ |
| `men_sparring_combat_6296158.mp4` | https://www.pexels.com/video/men-sparring-in-a-combat-sport-6296158/ |
| `woman_sparring_with_man_6296385.mp4` | https://www.pexels.com/video/a-woman-sparring-with-a-man-6296385/ |
| `man_woman_boxing_ring_4753947.mp4` | https://www.pexels.com/video/a-man-is-standing-in-a-boxing-ring-with-a-woman-4753947/ |

## Vandalism
| file | source |
|---|---|
| `man_vandalizing_glass_7120813.mp4` | https://www.pexels.com/video/man-vandalizing-a-glass-7120813/ |
| `people_vandalizing_wall_5561750.mp4` | https://www.pexels.com/video/people-vandalizing-the-wall-5561750/ |
| `man_vandalized_abandoned_building_7121131.mp4` | https://www.pexels.com/video/man-vandalized-an-abandoned-building-7121131/ |
| `person_vandalizing_7109134.mp4` | https://www.pexels.com/video/person-vandalizing-7109134/ |
| `person_vandalizing_wall_7108921.mp4` | https://www.pexels.com/video/person-vandalizing-a-wall-7108921/ |

## Arson
| file | source |
|---|---|
| `drone_fire_and_smoke_11584957.mp4` | https://www.pexels.com/video/drone-footage-of-fire-and-smoke-11584957/ |
| `drone_burning_establishment_8365993.mp4` | https://www.pexels.com/video/drone-footage-of-a-burning-establishments-8365993/ |
| `top_view_fire_and_smoke_11574636.mp4` | https://www.pexels.com/video/top-view-of-fire-and-smoke-11574636/ |
| `responding_fireman_6746452.mp4` | https://www.pexels.com/video/video-of-a-responding-fireman-6746452/ |
| `fire_and_emergency_services_11584958.mp4` | https://www.pexels.com/video/top-view-of-fire-and-emergency-services-11584958/ |

## Normal
| file | source |
|---|---|
| `pedestrians_crossing_antwerp_14413746.mp4` | https://www.pexels.com/video/pedestrians-crossing-the-street-in-the-city-of-antwerp-belgium-14413746/ |
| `pedestrians_toronto_14365420.mp4` | https://www.pexels.com/video/pedestrians-in-the-streets-of-toronto-city-in-ontario-canada-14365420/ |
| `pedestrians_crossing_street_27697943.mp4` | https://www.pexels.com/video/pedestrians-crossing-the-street-27697943/ |
| `street_in_toronto_19912796.mp4` | https://www.pexels.com/video/street-in-toronto-19912796/ |
| `traffic_control_intersection_4791196.mp4` | https://www.pexels.com/video/traffic-control-in-an-intersection-road-4791196/ |

## License
All clips are Pexels License (free for personal & commercial use, no attribution
required) — kept here for traceability, not because it's legally required.

## Kaggle download script (unused, blocked on auth)
`download_data.py` in the project root still targets the real
`sanskar457/ucf-crime` dataset via `kagglehub`. Once valid Kaggle credentials
are available, rerun it and replace this folder's contents with the actual
UCF-Crime clips.
