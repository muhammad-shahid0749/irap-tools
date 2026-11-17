# Evaluation definitions

This document defines what `irap_evaluation` computes when it scores a prediction file ([format](prediction_format.md)), and ends with how to [interpret the results](#interpreting-results). The definitions match the training-time evaluation of `vidlu_irap_gaim` (`get_irap_metrics`): on its model-compatible set, a model gets the scores reported during training, up to float32 rounding (`tests/test_vidlu_parity.py`).

Scoring a file on an evaluation set (`evaluate_predictions`) has these steps:

1. Select the segments and attributes ([evaluation sets](#evaluation-sets), [attributes](#attributes)).
2. Match the predictions to the labels, and score the invalid attribute predictions according to the null policy ([matching](#matching), [invalid attribute predictions](#invalid-attribute-predictions)). Attributes that the file does not predict are an error, or their attribute predictions count as invalid ([missing attributes](#missing-attributes)).
3. Accumulate statistics per road sequence ([statistics](#statistics)).
4. Compute the metrics from the summed statistics ([metrics](#metrics)).

The models of a method are then combined by their mean ([methods](#methods)). `compute_bootstrap_intervals` and `compare_methods` resample the road sequences and add the spread of a method's models ([confidence intervals](#confidence-intervals), [comparing methods](#comparing-methods)).

## Terms

- **Method** and **model**: a file holds the predictions of one model of a method, e.g. a network trained with one seed. Files with the same `model.method_name` are models of one method and are told apart by `model.seed` ([format](prediction_format.md#model-field)). A method with a single model needs no seed. The models of a method have the same training and early stopping splits (`check_method_models`), since a method is one training recipe.
- **Training split**: a split whose segments were used to fit a model (`model.training_splits`). Scores on it are not of held-out data. What a model used the scored split for (`ModelInfo.get_split_use`, `split_use`) is one of:
  - `training`: a training split,
  - `early_stopping`: used only for early stopping or checkpoint selection, so the scores are optimistic,
  - `unknown`: the training splits are unknown, so the model may have been fitted on it,
  - `held_out`: not used.

  `irap-eval evaluate` and `compare` print a note for the first three. `summary_long.csv`, `averages_*.csv`, `method_averages_*.csv` and the JSON documents of `irap-eval evaluate` have a `split_use` column or field.
- **Road sequence**: the segments of one road in driving order, one entry of `road_id_to_segment_id_sequence.json`.
- **Class index**: the position of a value in `attribute_value_to_irap_number` of `attribute_metadata.json`. It need not follow iRAP code order. Class 0 is the first value.
- **Attribute prediction**: the predicted distribution of one attribute for one segment. It is invalid if the model gave no usable prediction (null in the file, see [Invalid attribute predictions](#invalid-attribute-predictions)).

## Evaluation sets

For context offsets $o_1, \ldots, o_m$, the evaluation set of a split holds its segments that pass all of these filters (`irap_data.select_split_segments`):

1. **Image**: the segment has an entry in `segment_id_to_data_paths_rel.json`.
2. **Labels**: each iRAP code is mapped to its class index. A code is missing if it is absent, None, `'None'`, negative, or not a class of the attribute. Vietnam leaves the attribute unlabeled for that segment. BH drops the segment if any attribute is missing. Any other value, e.g. a float, is an error.
3. **Context**: for a segment at position $p$ of a sequence of length $L$, every $p + o_j$ lies in $[0, L)$, and the segment there has an image. Context segments need no label.
4. **N-context (BH only)**: the segment and the 10 segments on each side of it in its sequence are all in the same `seg_to_res/<split>.pickle` (train, val or test).

A file is scored on up to three sets:

- **Reference set**: offsets −4 … 0 for Vietnam and −10 … 10 for BH (`DATASET_PRESETS`). It is the same for every model, so its scores are comparable across models.
- **Full set**: offsets 0, i.e. every segment of the split with an image and labels (`get_full_set`). It is also the same for every model, and contains every other set. Only models that predict all its segments are scored on it, e.g. models without context or with padding for missing context segments.
- **Model-compatible set**: the model's own offsets, from the file header or `--context-offsets`. This is the set of the training-time evaluation. If the offsets lie within the reference window, the model-compatible set contains the reference set.

`select_evaluation_sets` (used by `irap-eval evaluate`) selects the sets as follows:

- A file that lacks segments of its model-compatible set is incomplete or of another metadata build, and is refused.
- A model whose offsets reach beyond the reference window lacks some reference segments. It is scored only on its model-compatible set, with a note, or refused if its offsets are unknown.
- A model that lacks segments of the full set is not scored on it, with a note.
- If the model-compatible set equals the reference set or the full set, it is left out.
- A set without labels, e.g. of an unlabeled split, is left out with a note. The file must still cover its segments as above.

## Attributes

The scored attributes are the canonical iRAP subset (`irap_data.attrs.get_attrs_to_include`, 43 attributes), in its order, minus those without a label in the evaluation set. On Vietnam validation, 35 remain. `--attributes all` starts from every attribute of the dataset.

### Missing attributes

A scored attribute that the file does not predict is handled according to `missing_attribute_policy` (`MissingAttributePolicy`, `--missing-attribute-policy`):

- **`error`** (default): scoring fails.
- **`invalid`**: every attribute prediction of the attribute counts as invalid ([invalid attribute predictions](#invalid-attribute-predictions)). All files are then scored on the same attributes, so their averages are comparable, and a method does not gain by leaving out attributes that it predicts poorly. `EvaluationResult.missing_attributes` (`missing_attributes` in the JSON documents) lists such attributes, and `num_invalid` counts their attribute predictions. With `first_class`, the attribute is scored as a constant prediction of class 0, which can score above 0, e.g. in accuracy if class 0 is common. With `exclude`, it would have no scored segments, which is an error ([invalid attribute predictions](#invalid-attribute-predictions)).

## Matching

- Every segment of the set must have a prediction. Extra predictions are ignored.
- Classes are matched by iRAP code. The file must predict exactly the dataset's codes of each attribute, in any order.
- The predicted class of an attribute prediction is the argmax of its distribution. On ties, it is the first maximum in dataset class order.

## Invalid attribute predictions

An invalid attribute prediction is a null in the file: the method gave no usable prediction of the attribute for the segment. The producer of the file decides what is usable. For VLMs, `vidlu_irap_gaim` (`is_usable_prediction`) marks an attribute prediction invalid if the response leaves out the attribute or names a value that matches none of its classes. Both cases are one category, since the boundary between them depends on the response parser. An invalid attribute prediction is not the same as:

- an unlabeled segment, which has no label and is not scored for the attribute,
- a missing row, i.e. a segment without a prediction, which makes the file incomplete (see [Evaluation sets](#evaluation-sets)).

The null policy (`NullPolicy`) selects how invalid attribute predictions are scored:

- **`first_class`** (default): in metrics computed from predicted classes, the invalid attribute prediction counts as a prediction of class 0, as in the training-time evaluation. In NLL and Brier, it counts as the uniform distribution, which gives $\ln K$ and $1 - 1/K$ for $K$ classes (a one-hot distribution of class 0 would give an infinite NLL for any other label).
- **`exclude`**: the invalid attribute prediction is not scored, as if the segment were unlabeled for the attribute. Each model leaves out its own invalid attribute predictions, so models with invalid attribute predictions are scored on different segments. An attribute whose attribute predictions are invalid on every labeled segment would have no scored segments and would drop out of the attribute averages, which would reward leaving it empty. Scoring it is therefore an error.

Means over models and comparisons therefore need `first_class`. `irap-eval evaluate` scores the models with `--null-policy` (default `first_class`) and writes the means over the models of a method (`method_averages_*.csv`) only with `first_class`. `irap-eval compare` always uses `first_class`.

The tables and documents have a `null_policy` column or field. In the library, `compute_method_metrics` and `compute_bootstrap_intervals` refuse several models scored with `exclude` if any of them has invalid attribute predictions, and `compare_methods` refuses models scored with `exclude` if any of them has invalid attribute predictions.

## Statistics

For each attribute and each road sequence $g$, the segments with a label for the attribute give:

- The confusion matrix $C^{(g)}_{ij}$: the number of segments with label $i$ and predicted class $j$.
- For `probs` files, the per-label sums
  $$s^{(g)}_{\mathrm{NLL},i} = \sum_{t:\,y_t=i} -\ln q_{t,y_t}, \qquad s^{(g)}_{\mathrm{Brier},i} = \sum_{t:\,y_t=i} \sum_k \left(q_{t,k} - [k = y_t]\right)^2,$$
  where $y_t$ is the label of segment $t$ and $q_t$ is its predicted distribution.
- For `probs` files, the confidence histogram $H^{(g)}$: for each class $k$ and confidence bin $b$, the numbers of segments in the bin that are predicted correctly as $k$, that have the label $k$ and another predicted class, and that are predicted as $k$ with another label. The confidence of a segment is the predicted probability of its predicted class, $q_{t,\hat y_t}$. The 50 bins are in the order of the confidence, and their edges are quantiles of the confidences of the attribute's labeled segments, so each bin has about 2 % of them. Equal confidences are in one bin.

The metrics of any set of sequences are computed from the summed statistics $C = \sum_g C^{(g)}$, $s = \sum_g s^{(g)}$ and $H = \sum_g H^{(g)}$. `EvaluationResult.statistics` keeps the per-sequence statistics, from which `compute_grouped_metrics` recomputes any metric without the predictions.

## Metrics

The metrics of an attribute are computed from $C$, with, for each class $k$:
- $\mathrm{TP}_k = C_{kk}$,
- the support $a_k = \sum_j C_{kj}$,
- the number predicted, $\hat a_k = \sum_i C_{ik}$,
- $n = \sum_k a_k$.

| Name | Definition |
|---|---|
| `P`, `R` | $P_k = \mathrm{TP}_k / \hat a_k$, $\;R_k = \mathrm{TP}_k / a_k$ (0 if the denominator is 0) |
| `F1` | $F_k = 2 P_k R_k / (P_k + R_k)$ (0 if $P_k + R_k = 0$) |
| `IoU` | $\mathrm{TP}_k / (a_k + \hat a_k - \mathrm{TP}_k)$ (0 if the denominator is 0) |
| `mP`, `mR`, `mF1`, `mIoU` | the mean over the classes $\mathcal{K} = \{k : a_k > 0 \text{ or } \hat a_k > 0\}$ |
| `A` | $\sum_k \mathrm{TP}_k / n$ |
| `n` | $n$ |
| `MCC` | $\dfrac{n \sum_k \mathrm{TP}_k - \sum_k \hat a_k a_k}{\sqrt{(n^2 - \sum_k \hat a_k^2)(n^2 - \sum_k a_k^2)}}$ (NaN if all labels or all predictions are one class) |
| `kappa` | $\dfrac{n \sum_k \mathrm{TP}_k - \sum_k \hat a_k a_k}{n^2 - \sum_k \hat a_k a_k}$ |
| `NLL`, `Brier` | $\sum_k s_k / n$ |
| `cNLL`, `cBrier` | $s_k / a_k$ per class (NaN if $a_k = 0$) |
| `mNLL`, `mBrier` | the mean of `cNLL`, `cBrier` over the classes with $a_k > 0$ |
| `MRP90` | the manual review proportion for `mF1` $\ge 0.9$, see [Manual review proportion](#manual-review-proportion) |

A class in $\mathcal{K}$ that is labeled but never predicted, or predicted but never labeled, has $F_k = 0$. Other undefined values (0/0) are NaN. `NLL` is infinite if a valid attribute prediction gives its label probability 0. The metrics of predicted distributions (`NLL` to `mBrier`, and `MRP90`) are not defined for `hard` files, and requesting them is an error.

### Manual review proportion

`MRP90` estimates how much of an attribute a reviewer still has to code. It simulates a review that goes through the segments from the least to the most confident prediction and replaces each prediction by the label. `MRP90` is the smallest proportion of the labeled segments that has to be reviewed for `mF1` to reach 0.9 (`MANUAL_REVIEW_TARGET_MF1`). It is 0 if `mF1` reaches 0.9 without a review. Lower is better.

- The target is `mF1`, not accuracy, because accuracy can reach a target while the rare classes are still wrong.
- `mF1` after a review never decreases, and is 1 when all segments are reviewed. A class that is only predicted, never labeled, leaves $\mathcal{K}$ when its last wrong prediction is corrected.
- An invalid attribute prediction counts as class 0 with the confidence $1/K$ of the uniform distribution, so it is reviewed early.
- `compute_review_curve` computes `mF1` and the accuracy after the review of the first $b$ bins of the confidence histogram, for $b = 0, \ldots, 50$, and `compute_manual_review_proportion` finds where the curve reaches the target, linear in the reviewed proportion between two points. The result differs from that of a review in the exact order of the confidences by at most the proportion of the segments in one bin, about 0.02 unless many confidences are equal. If all confidences are equal, it is the manual review proportion of a review in random order, $(0.9 - \mathrm{mF1}) / (1 - \mathrm{mF1})$. The curve's reviewed proportions are `review_proportions` of `ReviewCurve` and of the JSON report's `review_curve`.
- The simulation assumes that the reviewer codes as the labels do. It says how well the confidences find the segments that lower `mF1`, not how a reviewer would decide.
- The idea is that of selective classification, where predictions with a low confidence are deferred to a person, and its risk–coverage curves: `MRP90` is about 1 − coverage at the target. Unlike the selective risk, which is computed on the kept predictions only, `mF1` here is computed over all segments, with the reviewed ones corrected.

**Support threshold.** The suffix `_suppN` restricts a mean over classes to $\mathcal{K}_N = \{k : a_k \ge N\}$, e.g. `mF1_supp10`. `nc_suppN` is $|\mathcal{K}_N|$.

**Attribute average.** The prefix `a` averages a metric over the scored attributes where it is not NaN, e.g. `amF1`. Infinite values are included. Each attribute counts equally, so `aNLL` is the mean of the per-attribute NLLs. Per-class metrics are not averaged.

`describe_metric(name)` describes any such name in words, e.g. for a tooltip.

**Averages over a subset of the attributes.** Since an average is the mean of the per-attribute values, `select_metric_attributes(result.metrics, subset)` computes it for a subset from those values, without the predictions. It needs the per-attribute value of each average, e.g. `mP` for `amP`. `evaluate_predictions` computes them by default, and `add_per_attribute_metric_names` adds them to other lists of metric names. Intervals of such averages cannot be computed from the per-attribute intervals, since the attributes of a road sequence are correlated. `select_result_attributes(result, subset)` restricts a result to the subset with its statistics, and `compute_bootstrap_intervals` and `compare_methods` then work on it as on any result.

The iRAP protocol (`get_irap_metric_names`) reports the following. Its main metric is `amF1`.

- **Attribute averages**: `amF1`, `amP`, `amR`, `aMCC`, `aA`, `amF1_supp5` and `amF1_supp10`. `probs` files also get `aNLL`, `aBrier`, `amNLL`, `amNLL_supp5` and `amNLL_supp10`.
- **Per attribute**: `mF1`, `A`, `n`, `MCC`, `mF1_suppN` and `nc_suppN`. `probs` files also get `NLL`, `Brier` and `mNLL`.

By default (`get_default_metric_names`), `evaluate_predictions` (and so `irap-eval evaluate`) also computes, for `probs` files, the manual review proportion `aMRP90` and `MRP90`, which is not part of the protocol, and the per-attribute values of the averages that the protocol lacks: `mP`, `mR` and, for `probs` files, `mNLL_suppN`.

Results keep only the attribute averages and per-attribute metrics. `compute_class_metrics(result)` computes the per-class metrics (`P`, `R`, `F1`, `IoU`, `cNLL`, `cBrier`), support and confusion matrix of each attribute from its statistics, and `compute_review_curves(result)` the review curve of each attribute.

## Methods

The value of a metric for a method with models $i = 1, \ldots, M$ on the same evaluation set is the mean $\bar m = \frac{1}{M} \sum_i m_i$ of the models' values (`compute_method_metrics`, or `compute_mean_metrics` for the metric values alone, e.g. over a subset of the attributes). The models must have the same segments, sequences, attributes, metrics, null policy, and training and early stopping splits. A method with a single model has the values of that model.

## Confidence intervals

The intervals cover two sources of variation (`--bootstrap B`, `--seed`, default 0):

- **Road sequences.** Segments of a road sequence are correlated, so the bootstrap resamples sequences, not segments.
- **Models.** Models of a method differ, e.g. by training seed. Their mean is uncertain, and more so with few models.

`compute_bootstrap_intervals(models)` computes them as follows:

1. Each of the $B$ resamples draws the $G$ sequences of the set $G$ times with replacement. This gives multiplicities $w \sim \mathrm{Multinomial}(G; 1/G, \ldots, 1/G)$. The draws depend only on the seed, $B$ and $G$, so all models on an evaluation set share the resamples.
2. All metrics of each model, including the means and averages, are recomputed from $\sum_g w_g C^{(g)}$, $\sum_g w_g s^{(g)}$ and $\sum_g w_g H^{(g)}$, giving $m^*_i$ (`resample_model_metrics`). The confidence bins are those of the whole set in every resample. A sequence with weight 0 adds nothing, also if its NLL sum is infinite. Averages over a subset of the attributes can be computed from the resampled per-attribute values (`select_metric_attributes`), as from the values themselves.
3. The resampled value of the method is
   $$v^* = \bar m^* + \frac{\mathrm{sd}(m^*_1, \ldots, m^*_M)}{\sqrt M}\, t^*, \qquad t^* \sim t_{M-1},$$
   with the sample standard deviation and one Student-t draw per resample, shared by the metrics of the method (`combine_model_resamples`). This is the posterior of the mean of normally distributed model values under the prior $p(\mu, \sigma) \propto 1/\sigma$. The t draws depend on the seed and the method name, so they are independent of the resamples and of the draws of other methods. With one model, $v^* = m^*_1$, and the interval covers only the road sequences. If $\bar m^*$ is infinite, $v^*$ is $\bar m^*$.
4. The interval at confidence $c$ (default 0.95, `compute_percentile_interval`) runs from the $(1-c)/2$ to the $(1+c)/2$ quantile of the $v^*$, with linear interpolation, as `numpy.quantile`. NaN values are left out, e.g. an MCC that is undefined in a resample, or the difference of two infinite values in a comparison. The interval then holds only for the resamples where the value is defined. Their number is reported (`BootstrapInterval.num_undefined`, `ci_num_undefined` in `summary_long.csv`, `num_undefined` in the JSON documents, the `undefined` column of `irap-eval compare`), and `irap-eval` prints the intervals that leave out some, but not all, resamples. Infinite values are kept, e.g. the NLL of a resample in which a label has probability 0. A bound interpolated between an infinite value and a finite or equal value is that infinite value.

Intervals are computed for every metric of a result. Per-class metrics have no intervals. The same seed gives the same draws for a method on the same set.

## Comparing methods

`compare_methods(models_a, models_b)` (`irap-eval compare --a A --b B`, on the reference set, default $B = 1000$) recomputes the models of both methods on the same resamples, which pairs the comparison over road sequences (`compute_resampled_differences`):

- The point difference is $\Delta = \bar m_a - \bar m_b$.
- Its interval is the percentile interval of the resampled differences $v^*_a - v^*_b$. The model terms of $a$ and $b$ use independent t draws (step 3 above).

There are no p-values. A difference is significant at level $1 - c$ if its interval excludes 0. A one-sided test at level $\alpha$, e.g. whether $a$ is worse than $b$, uses the interval at $c = 1 - 2\alpha$.

All models of both methods must have the same segments, sequences, attributes and null policy.

## Interpreting results

- **Base conclusions on differences between methods on the same set.** Macro metrics (`mF1`, `_suppN`) depend on which classes the evaluation set contains, so a method's interval is not an interval for a population value.
- **`amF1` is the main metric.** Differences per attribute or for many metrics are exploratory, since some will exclude 0 by chance.
- **The intervals generalize to roads like those of the evaluated region, and to further models of the methods.** The splits are by region, so they say nothing about other regions. For one region of a set, use `select_evaluation_subset`.
- **Scores on a split that a model used** (`split_use`) are not of held-out data: those on a training split are not comparable with held-out scores, and those on an early stopping split are optimistic.
- **Invalid attribute predictions.** With `first_class`, every model is scored on the same segments, but an invalid attribute prediction counts as class 0, which is often, but not always, the most common class. With `exclude`, the scores measure skill when the response is usable, and must be read together with the number of invalid attribute predictions (`num_invalid` in the JSON documents).

Simulations on the Vietnam labels with synthetic predictors (2026-10-01, about 200 repetitions each) support the design of the intervals:

- **Model variation is not negligible.** With 3–5 models per method, intervals of $\Delta$ amF1 over road sequences alone contained the true difference in only 18–55 % of the repetitions. With the t term, they did so in 97–100 %, so the intervals are conservative. A bootstrap over the models, instead of the t term, reached only 80–89 %, since 3–5 values say little about their spread.
- **Single models.** Intervals of the difference of two single models covered the true difference in 85–96 %.
- **Single methods.** 95 % intervals of a single method's amF1 contained the value on all roads in 69–90 % of the repetitions, and those of `amF1_supp10` in 0–68 %.
- **Rejected alternatives.** Dirichlet weights (the Bayesian bootstrap) and bias-corrected intervals were not better, and in some cases worse.
