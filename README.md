# Freeway traffic estimation

A change that improved my held-out state reconstruction moved the real score down. The low-rank matrix completion I added in the fourth stage beat the interpolation baseline on a time-disjoint month of train data, then scored 0.81161 against 0.81188 on the hidden validation month. A bounded version of the same correction, half the size and clipped, scored 0.81652. The model was right on average and wrong in the tails, with speed extrapolations as high as 272 km/h, and the score punished the tails.

That pattern repeated at most stages of this project, and the rest of the repository is built around it: local proxies that were cheap to compute, a faithful local rebuild of the one scorer I could rebuild, and a habit of treating every proxy gain as a hypothesis.

## The problem

TrafficFlowBench is a synthetic freeway benchmark: ten directional panels (five corridors, both directions) at 5-minute resolution, with mainline detectors, ramp counts, link geometry, a triangular fundamental diagram and a weak origin-destination prior. Four tasks are scored together:

```
S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
```

| task | what is predicted | notes |
|---|---|---|
| State | speed and flow in blanked detector cells | RMSE, scaled by 25 km/h and 600 veh/h/lane |
| Queue | queued or not, 5 to 30 minutes ahead, per link | space-time IoU; labels withheld |
| Physics | none directly | scored from the state answer: fundamental-diagram fit and vehicle conservation |
| OD flow | non-negative path flows | regularised non-negative least squares against link counts |

Only the state task can be scored locally in full. Queue labels are withheld, and the released physics scorer depends on boundary flows that are not published.

## Results

Composite score on the held-out validation month, by stage:

| stage | change | score |
|---|---|---:|
| v1 | same-day temporal and neighbouring-link interpolation; recurrent-bottleneck onset; ODME with NNLS | 0.8026 |
| v2 | queue rules from train events | 0.8111 |
| v3 | downstream propagation of ongoing queues on one corridor | 0.8119 |
| v4 | low-rank matrix completion for state | 0.8116 (worse) |
| v5 | half of the correction, clipped to +/-5 km/h and +/-50 veh/h/lane | 0.8165 |
| v6 | 75% of the clipped correction | 0.8174 |
| v7 | onset location conditioned on time of day (Gaussian kernel over train onset events) | 0.8297 |
| v8 | gradient-boosted ongoing queue model | 0.8234 (worse) |
| v10 to v15 | gradient-boosted state residuals; pooled and per-panel queue models; onset priors; bagging | 0.8415 to 0.8586 |
| v20 to v27 | whole-corridor 1-D CNN for ongoing queues; whole-day 2-D CNN for state residuals | 0.8619 to 0.8660 |
| v30 to v49 | blend-weight search on state | 0.8667 |
| v66 to v71 | three-model onset ensemble, extra history context | 0.8688 |
| v72 to v79 | blend with published gradient-boosted models; final weights chosen by interpolation | 0.8770 |

The last two rows bring in other people's models (see `NOTICE`). The code in `src/` is my own pipeline up to and including the CNN stage and the onset ensemble; the final blending step is described in [`docs/findings.md`](docs/findings.md) but not included, because its largest component is not mine.

## What I found

**Decomposing the score told me where not to work.** I replaced one task's output at a time with a deliberately uninformative answer and read the change in the composite. An all-zero queue answer implied `S_queue` = 0.698. An invalid path-flow answer implied `S_ODME` = 0.994. ODME was saturated, so none of the remaining effort went there, and queue forecasting held up to 0.09 of composite headroom. Nearly everything after v8 is queue and state work.

**The physics score can be rebuilt on train, and then it matches.** The organizers' conservation check uses withheld boundary flows that are projected onto the released observations, so on train, where unmasked observations are released, the projection can be reconstructed. `src/physics_local.py` does that: an oracle answer scores `S_LWR` = 1, and the local score deltas for v5 and v6 (+0.0056 and +0.0068) track the real ones (+0.0046 and +0.0055). The v4 delta does not match; I suspect the 90-minute all-link gaps that follow each queue window, where interpolation is least reliable, but I did not isolate it.

**Proxy labels hid selection effects.** Queue labels are derived from noisy released observations, not the latent state the real scorer uses. The learned ongoing-queue model at v8 raised proxy IoU on first-of-day windows from 0.617 to 0.850 on a month held out from fitting and threshold selection, and still lost 0.0063 on the real score. Later, a demand-feature onset model gained 0.017 IoU on the June windows and lost 0.0014 on the real score. After that, a proxy gain was a reason to test a change, never a conclusion.

**Bounding a good model beat tuning it.** The matrix-completion model passed every local test. Clipping its correction and taking half of it was worth +0.0046 on the real score, against -0.0003 for the unclipped model. The lesson I kept is to cap a learned correction's worst case before measuring its average.

**Blend weights set by interpolation, not by a sweep.** For the last state blend I had three real score readings at weights 0, 0.5 and 0.75, and the middle one was best. A quadratic through them put the maximum near 0.42; I used 0.40. That is a fit to three noisy points, labelled as such in the report, and it moved the score by 0.00005.

## Repository

```
src/state*.py, build_state*.py     state reconstruction: interpolation, low-rank completion, bounded blend, GBM and CNN residuals
src/queue*.py, build_queue*.py     queue forecasting: recurrent bottlenecks, event rules, time-conditioned onset, GBM and CNN models
src/onset_v66.py                   onset ensemble with demand features
src/odme.py                        regularised NNLS path-flow estimation
src/physics_local.py               local rebuild of the conservation score on train
src/assemble.py, validate_submission.py   strict streaming assembly and full-key validation of the six-column answer file
src/bench_*.py, queue_eval.py      time-disjoint backtests
config/                            frozen per-corridor selections (rank, blend, thresholds, bandwidths)
tests/                             unit tests for the core modules
docs/findings.md                   stage-by-stage notes and the cases where local gains did not transfer
```

Stage numbers in file names follow the table above.

## Running it

```bash
pip install -r requirements.txt
git clone https://github.com/jacky850/trafficflowbench-public organizer-code   # organizer's MIT-licensed scorers and schemas
# place the released data under data/release/
python -m src.build_state && python -m src.build_queue && python -m src.odme
python -m src.assemble && python -m src.validate_submission
python -m pytest -q
```

Memory is the constraint: the full release is large, so everything runs panel by panel on an 8 GB machine.

## License

Apache-2.0 for my code. See `LICENSE` and `NOTICE`.
