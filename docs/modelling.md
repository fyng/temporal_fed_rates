# What moves the Fed

The best guide to what the Federal Reserve will do next is what it has just done. Once you know the recent path of interest rates, inflation, unemployment and financial stress add nothing to a forecast, and the policy rules the Fed calculates for its own meetings add little.

We did not set out to show this. We built two unrelated models, one of the level of the Fed's target rate and one of the direction of each decision. They disagree about nearly everything except that conclusion.

## The record

The Federal Open Market Committee, the Fed's rate-setting body, made 271 decisions between February 4th 1994 and September 16th 2026. Two-thirds were holds.

| Decision | Count | Share |
| --- | --- | --- |
| Cut, 50bp or more | 19 | 7% |
| Cut, 25bp | 21 | 8% |
| Hold | 179 | 66% |
| Rise, 25bp | 41 | 15% |
| Rise, 50bp or more | 11 | 4% |

(A basis point, bp, is a hundredth of a percentage point.)

That 66% is the central difficulty. A model that always says "hold" is right two times in three, so accuracy tells you little. We therefore use scores that give the rare decisions their due:

- **Weighted kappa** measures agreement with the actual decisions beyond what chance would give, and punishes a miss more the further it lands from the truth.
- **The ranked probability score** rewards a model for putting its probability on the right outcome and on outcomes near it. Lower is better.
- **The area under the ROC curve** asks, for each kind of decision, how often the model ranks a meeting that took it above one that did not. Always saying hold scores 0.5.
- **The area under the precision-recall curve** asks the same question but punishes false alarms on rare decisions harder. Always saying hold scores the share of each decision, 0.2 on average.
- **F1** balances how many of each decision the model catches against how many of its calls are right.

The last three are computed for each of the five kinds of decision against the rest and then averaged, so a 25bp cut counts as much as a hold.

We also lag every input by a month. The committee sets rates knowing last month's figures, not this month's: consumer prices for one month appear only in the middle of the next. Matching each meeting to data dated the same month hands the model figures that did not yet exist.

## Model one: the level of the rate

The first model is the standard one. The Fed picks a target for its rate based on inflation and slack in the economy, then moves only part of the way towards it at each step:

```text
i_t = rho * i_(t-1) + (1 - rho) * (r* + pi_t + a*(pi_t - 2) + b*gap_t) + e_t
```

Here `rho` measures inertia, `a` the response to inflation above 2% and `b` the response to slack. We estimate it on quarterly data from 1983, with standard errors robust to the correlation between neighbouring quarters.

| | Estimate | Standard error |
| --- | --- | --- |
| rho, inertia | 0.934 | 0.023 |
| a, inflation response | 1.04 | 0.78 |
| b, slack response | 1.69 | 0.58 |

n = 175, R² = 0.972.

**Inertia and the response to slack are clear. The response to inflation is not.** It has the right sign and roughly the size John Taylor proposed, but lies just 1.3 standard errors from zero. More care with the data will not fix this. The model recovers `a` and `b` by dividing by 1 − rho, and as rho nears one the divisor shrinks and takes the precision with it. The coefficients estimated before that division are all tight: 0.934 (0.023) on the lagged rate, 0.135 (0.056) on inflation and 0.112 (0.030) on slack.

![Signal failure](../figures/production/specification_grid.png)

Shortening the sample makes things worse:

| Sample | Monthly | Quarterly |
| --- | --- | --- |
| 1961 on | rho .969, a 0.37 (.61) | rho .910, a 0.38 (.47) |
| 1983 on | rho .979, a 1.37 (.84) | rho .934, a 1.04 (.78) |
| 1994 on | rho .987, a **4.01** (3.67) | rho .943, a 1.50 (1.95) |

The monthly estimate from 1994 implies that the Fed raises rates by four points for every point of excess inflation. That is not a finding; it is division by almost zero. Monthly data make the problem worse. The Fed meets eight times a year and usually holds, so most months show no change and rho is pushed towards one. We report quarterly estimates, as Taylor, Richard Clarida, Jordi Galí, Mark Gertler and Glenn Rudebusch did.

**The model finds one real break.** Across rolling ten-year windows ending before 1983, the median response to inflation is −0.61. For windows ending between 1983 and 1999 it is +0.78, and the estimate holds steady. This reproduces the result of Clarida, Galí and Gertler on our data: the Fed did not respond to inflation before Paul Volcker, and did afterwards.

After 2000 the estimate falls apart. Median rho climbs from 0.93 in the 1983-99 windows to 0.98, and the division by 1 − rho throws 54% of windows ending since 2000 off the chart's scale, some as far as ±560. Since 2000 the rolling windows cannot pin down the Fed's response to inflation at all.

![The Volcker break](../figures/production/rolling_inflation_response.png)

**Estimates for individual chairmen fail.** For every chairman since Alan Greenspan, rho comes out at or above one and the division blows up: Jerome Powell's inflation response is −20.2, with a standard error of 66. Only Greenspan's 221 months yield usable figures: rho of 0.949, `a` of 0.75 and `b` of 2.38. The code declines to report the others.

**Against a random walk, the model ties.** Forecasting one month ahead, its root-mean-square error is 0.548 against 0.539 for simply assuming no change, a ratio of 1.015. It wins only after 2000, with a ratio of 0.982. That is almost built in: a rate that does not move in most months is nearly a random walk over a month. We report it to be clear that the model barely beats a naive guess.

## Model two: the direction of each decision

The second model predicts which of the five kinds of decision the committee will take, using an ordered logit, a regression suited to ranked outcomes. We test it strictly forward in time: we predict each meeting from the meetings before it alone, refitting every time. That gives 171 test meetings, from December 2005 to September 2026. Shuffling the data for cross-validation, a common shortcut, would leak the future into the past.

The baseline is the share of each kind of decision among the meetings before each one, which uses no future data either.

| Model | Inputs | Kappa | Probability score | ROC area | PR area | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **History only** | **3** | **0.64** | **0.052** | **0.83** | **0.50** | **0.50** | **78%** |
| History + rule gaps | 8 | 0.61 | 0.058 | 0.86 | 0.52 | 0.50 | 73% |
| History + economy | 14 | 0.58 | 0.073 | 0.79 | 0.43 | 0.49 | 70% |
| Everything | 19 | 0.55 | 0.079 | 0.82 | 0.43 | 0.41 | 68% |
| Rule gaps only | 5 | 0.35 | 0.082 | 0.80 | 0.42 | 0.31 | 68% |
| Economy only | 11 | 0.32 | 0.109 | 0.60 | 0.31 | 0.31 | 64% |
| *Past shares* | — | *0.04* | *0.084* | *0.56* | *0.25* | *0.22* | *72%* |
| *Always hold* | — | *0* | *0.092* | *0.50* | *0.20* | *0.17* | *73%* |

Rule gaps are the distance between the actual rate and what each of the Fed's published rules prescribes.

![Past performance](../figures/production/decision_ablation.png)

The three history inputs are the previous decision, the number of meetings since the rate last moved, and whether the meeting was unscheduled. With them alone, the model's probability score is 34% lower than the 19-input model's and 38% lower than the baseline's. It doubles the baseline's precision-recall area and more than doubles its F1.

**Adding economic data makes the model worse on every score.** Alone, they barely beat the baseline at ranking decisions and score worse than it on probabilities, 0.109 against 0.084. Added to history, they lower every score.

**The Fed's own rules are a closer call.** Alone, the rule gaps rank decisions well, with an ROC area of 0.80, but their probability score is no better than the baseline's. Added to history, they lift the ROC area from 0.83 to 0.86 and the precision-recall area from 0.50 to 0.52, while worsening the probability score from 0.052 to 0.058 and leaving F1 unchanged. The rules help the model order meetings slightly but make its probabilities less reliable. On 171 meetings, differences this small could be noise.

Accuracy would hide all this. Always saying hold is right 73% of the time; the history model manages 78%. The 19-input model, at 68%, is less accurate than always saying hold, yet scores far better on every other measure, because it catches some of the rarer decisions that the hold rule never does.

The model does not need constant refitting. Refit only every eighth meeting, about once a year, it scores 0.0542 against 0.0523, still far ahead of the baseline. Its edge comes from inputs that track the Fed's latest moves, not from re-estimating the model.

## What we could not predict

**The model never correctly calls a 25bp cut: it scores nought out of eleven.**

The obvious story fits the facts. Small cuts are insurance, made in the middle of a cycle, at scheduled meetings, with the jobs market still strong. Compared with big cuts, small ones come as payrolls grow by 86,000 a month rather than shrink by 81,000, with the VIX, a gauge of expected stockmarket turbulence, at 21 rather than 28. Just 5% come at unscheduled meetings, against 37% of big cuts, and 19% during recessions, against 42%. Some of these gaps reach 0.85 of a standard deviation.

**Yet the two kinds of cut still overlap.** Assigning each cut to whichever group's average it sits nearer, leaving it out of that average, sorts 62.5% of them correctly, against 47.5% by guessing the more common kind. That figure barely moves whichever inputs we use.

![Insurance claims](../figures/production/cut25_misses.png)

The errors point elsewhere. Of the eleven misses, the model calls eight holds and only three big cuts, and it never gives a small cut more than a 29% chance. Its ranking is not hopeless: for small cuts its precision-recall area is 0.24, against 0.07 for the baseline. It puts them above chance, but never on top. It can tell small cuts from big ones well enough. What it cannot do is see a precautionary cut coming, because on the day such meetings look like the holds either side of them. Nothing in the published data flags an insurance cut in advance. We tuned nothing to these eleven cases.

## What this does and does not show

It does not show that the Fed ignores the economy. Both models are simple statistical fits, and inertia absorbs everything persistent, including the slow-moving economic conditions that took the rate to where it already is. A committee that responded quickly and fully to the economy could produce a rate series much like this one.

It does show that, to predict the Fed's next move, its recent behaviour tells you nearly everything the published data can. Economic data add nothing beyond it, and the policy rules the Fed calculates for its own meetings add little.

## Caveats

- The neutral interest rate, r\*, is fixed at 2%. A time-varying estimate from Thomas Laubach and John Williams, later with Kathryn Holston, is available in the code but untested in these models.
- Records of dissenting votes start in March 2002. Earlier dissents sit only in the minutes, so dissent is not an input.
