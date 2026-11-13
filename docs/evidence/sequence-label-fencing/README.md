# BiLSTM label-knowledge fencing

Reproduced against development commit33378c9ef55c87cf51bf4896524ba679325a1dbf:
when every training label revision was first knowable at the dataset research
cutoff, the sequence adapter nevertheless fitted6models and returned72
out-of-sample predictions. Its raw-target fallback bypassed revision knowledge.
The controlled dataset is a labeled SYNTHETIC fixture, not observed-market evidence.

The repair rejects unavailable supervised labels before actual fitting and before
any research persistence. Feature-only warm-up observations and label-end windows
already purged by the canonical sequence kernel retain their existing behavior.
Known valid runs preserve their fitted model identities and predictions.

Loading an older immutable BiLSTM run rechecks its original admitted dataset,
training prefix, embargo exclusions and label knowledge. Unsafe legacy runs fail
closed before inference, cached section publication or allocation. Their files
are preserved, not rewritten. Runtime budgets, lease fencing, broker restrictions
and model-promotion controls are unchanged.

Focused acceptance:53passed,0failed,0skipped (new regression tests plus persisted
models, numerical sequence, forests and financial retrieval). Tests cover rejection
before the actual fitter/persistence, irrelevant warm-up target changes, legacy
immutable artifact preservation, safe HTTP422 and unchanged prior revision tests.
Full regression, isolated wheel and final-head CI results follow in this directory
and the pull-request evidence.

This is a research correctness repair, not profitability or certification evidence.
fully_funded_certified=false; T4_NOT_CERTIFIED; historical§139UNSUPPLIED.
