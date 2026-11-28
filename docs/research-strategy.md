# Offline research strategy simulation

`POST /api/v1/workspace/research-strategy` builds an allocation from a saved
research screen, prepares durable offline intents, and optionally fills them
in the local OMS at the allocation reference prices.

- `operation` defaults to `simulate`; `replay` reloads a saved artifact
- `fill: true` dispatches and fills prepared client ids offline
- `trading_approved` stays false; `broker_connected` stays false
- Fills are not market observations and do not enable live brokerage

Broker connectivity remains disabled.
