# Saved allocation research-lineage fencing

A follow-up at891231051ae5ae94df3fa27da192c30b7ed97081 reproduced a legacy
allocation gap. A saved plan from the former unknowable-label BiLSTM run could
still prepare2offline simulated intents because allocation loading verified its
own hash but did not revalidate the original research source. This was an
authored SYNTHETIC fixture with broker_connected=false, not broker activity.

Saved allocations now revalidate their original model, signal or pair research
source and matching dataset before cached views or preparation. Signal loading
already verifies its constituent model runs. Rejection occurs before opening the
OMS database, including before returning a cached preparation. Immutable evidence
and historical executions are not rewritten or reversed.

27focused tests pass, covering the reproduced unsafe plan, no OMS persistence on
rejection, immutable plan preservation, normal signal/pair allocation and restart
workflows. This extends the earlier label-knowledge repair; its1,270-test
acceptance retains its exact earlier source scope. Updated complete regression,
current installed wheel and final-head CI acceptance are running for this
follow-up. No model family, broker connectivity, promotion or new service added.
