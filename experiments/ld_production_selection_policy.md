# LD production-method selection policy

**Prospective status:** fixed before the component-uniform candidate is trained
or any of its outer-test scores are inspected. The binary-label rule, grouped
five-fold map, frozen baseline, and the separate [candidate protocol](ld_component_uniform_sampling_protocol.md)
remain unchanged.

The five-fold candidate procedure makes an independent inner-validation gate
decision for each outer fold. Those five decisions are not themselves a single
model that can be trained on all 36 images. For a final all-36 model, use this
conservative, validation-only method rule:

1. Use the component-uniform method only if its predeclared gate selects it in
   at least **four of five** folds.
2. The mean of all 36 paired *inner-validation* fixed-0.50 Dice differences
   (candidate minus baseline, one validation appearance per image) must be at
   least **+0.01** absolute. Include all folds, even ones whose gate failed.
3. Across each of the five validation folds, mean fixed-0.50 precision may not
   fall by more than **0.03** absolute, and no individual validation image may
   lose more than **0.10** absolute Dice. Apply these safety limits to all
   folds, not only the winning ones.

If any condition fails, retain the frozen LD baseline as the final method.
Do not use outer-test Dice, IoU, visual appearance, or worst-case OOF images
to override this decision. Do not change the rule after seeing results.

For the selected method, derive final training duration as the median of the
five *earliest best full-image validation-Dice epoch counts* and deployment
threshold as the median of all five validation-selected thresholds. Use all
five candidate folds if the candidate is selected, including gate losers; do
not cherry-pick favorable folds. Train fresh weights on all 36 derived labels
for the fixed number of epochs, with no validation-driven tuning during that
final fit. Save a new model, policy, code/data hashes, and a separate output
directory. The baseline final-fit entry point must not be reused unchanged
for a candidate because it uses pixel-uniform positive-center sampling.

These numeric margins are pragmatic stability guardrails, not significance
tests. Rotated inner validation exposes every image that is outer test in
another fold, so a global production-method decision using all five
validation sets prevents treating a single candidate's same-dataset OOF
summary as an untouched confirmatory estimate. Report the predeclared
fold-local gated OOF as *exploratory internal evidence* and validate the
deployed single model on future, independently labelled organoid images.
