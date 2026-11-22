# Findings, stage by stage

Competition scores here are the composite on the public validation month. Final ranking uses a different private month. Local numbers are time-disjoint backtests on train: January selects, February checks. Queue labels used locally are derived from noisy released observations, so they are proxies for the latent labels the real scorer uses.

## Where local and real disagreed

| stage | local evidence | real result | what I took from it |
|---|---|---|---|
| v4 low-rank state completion | better on every corridor on held-out February | 0.81161, down from 0.81188 | tails matter more than the mean; extrapolations reached 272 km/h |
| v5 bounded correction | 60 of 60 field, corridor and regime groups improved under an artificial-masking stress test | 0.81652 | clip the worst case first |
| v8 learned ongoing queues | first-of-day IoU 0.617 to 0.850 on February | 0.82342, down from 0.82972 | proxy labels reward fitting the observation noise |
| v67 demand-feature onset | June IoU 0.914 to 0.931 | 0.86685 against 0.86826 | a gain on 40 windows is not a gain |
| v74 calibrated ongoing blend | large local ongoing gain | 0.87113 against 0.87517 | same pattern, third time |
| v80 full gap-trained LightGBM state | best of the new models on corrected local validation (S_state 0.9345) | 0.87642 against 0.87692 | the baseline it was compared with was not the component it replaced |

## A silent date bug

A validation cache written by Pandas 3 stored timestamps as microsecond integers, and the reader interpreted them as nanoseconds. Every aggregate computed from that cache carried wrong dates. The production features and models used ISO dates and were unaffected. I found it because a vectorised scorer disagreed with the per-day reference scorer; after the fix, the two agreed to 9e-10 on all 28 days and three masking regimes for one corridor. The reader now asserts that the year is 2031. Any model-selection number computed before the fix was discarded.

## Forecast-origin rule

Queue windows give 60 minutes of history ending at T-5, and the scorer forbids anything timestamped after T for that window, including ramp counts inside the horizon and observations that resume after the blanked buffer. The organizers confirmed that an observation timestamped exactly at T is allowed. I tested T-augmented onset features against history-only features with exact-timestamp assertions. They did not beat the proxy baseline and were left out.

## Last stage: blending with published models

The final five evaluations changed only the state component, and each file passed full 6,980,503-row key validation plus an independent comparison against the previous best:

| change | score |
|---|---:|
| half-weight gradient-boosted gap state | 0.87692 (unchanged) |
| half-weight density correction to flow | 0.87648 |
| 75% published state on non-gap rows | 0.87589 |
| full LightGBM gap replacement | 0.87642 |
| 40% published state on non-gap rows | 0.87697 |

Every new trained model lost. The only gain came from the blend weight, by 0.00005.
