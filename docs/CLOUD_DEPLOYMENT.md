# Cloud deployment

Checked against vendor docs on 2026-10-08. Nothing in this repository was provisioned. No billing account was enabled. Free tiers change, and several require a credit card even when the VM itself is listed at $0.

The laptop is the terminal. A small VM can run `gig serve` and a scheduled ingest. Training, large backtests, and any future FinBERT or BiLSTM job do not belong on that same process.

## 1. Recommended free-cloud architecture

One Always Free Ampere A1 VM (Oracle Cloud Infrastructure), shaped inside the always-free envelope:

- 2 OCPU and 12 GB RAM as a single `VM.Standard.A1.Flex`, or split only if the paper loop and a research queue must be separate processes on separate instances that still sum to 2 OCPU and 12 GB.
- Ubuntu ARM64, Python 3.12 from the OS, venv, `pip install -e ".[dev,web]"`.
- `gig serve` bound to localhost. Reach it through SSH local forwarding from the laptop (`ssh -L 8000:127.0.0.1:8000`). Do not open port 8000 on a public IP; the API has no authentication.
- DuckDB on the VM disk. Snapshot the volume before upgrades.
- Research jobs as a manual or cron process with a timeout, not inside the uvicorn worker.

This is not high availability. An Always Free instance can be stopped for capacity, and a reboot must reload state from DuckDB and from the broker before any new paper order.

GCP `e2-micro` is the fallback if an Oracle account is unavailable. It is too small for the HMM/GARCH diagnostic plus a factor cloud on a 200-name panel. Use it only as an API that refuses heavy routes.

## 2. Provider comparison

| Provider | Offer (as published) | CPU / RAM | Storage | Suitability for an always-on paper loop | Billing risk |
|---|---|---|---|---|---|
| Oracle Cloud Always Free | First 1,500 OCPU-hours and 9,000 GB-hours per month of `VM.Standard.A1.Flex`, described as 2 OCPU and 12 GB for an Always Free tenancy. Up to two `VM.Standard.E2.1.Micro` (AMD, 1/8 OCPU, 1 GB). 200 GB block volume. Home region only. South Korea North (Chuncheon) cannot host Always Free A1. | 2 OCPU / 12 GB if the A1 capacity is granted | 200 GB, boot volume minimum about 47 GB | Best fit of the $0 options, if the region has capacity | Trial upgrades and shapes above 2 OCPU or 12 GB are billed. A card is required at signup. Idle capacity is not guaranteed. |
| Google Cloud Free Tier | 1 non-preemptible `e2-micro` per month in `us-west1`, `us-central1`, or `us-east1`, measured in hours across those regions. 30 GB-months standard disk. 1 GB North America egress. GPUs are not included. | shared-core `e2-micro`, 1 GB memory | 30 GB-month disk | Marginal. 1 GB will OOM a panel fit. | A billing account exists. Usage past the free envelope is invoiced. External IPs and other regions cost money. |
| AWS | New accounts: up to $200 credits over 6 months (Free plan closes when credits end or at 6 months). EC2 is credit-eligible, not an always-free VM. Older accounts (before 2025-07-15) had a 12-month t-class allowance. | whatever the credit covers | EBS is extra once credits are gone | Fine for a short experiment. Not a permanent host. | Free plan versus paid plan is a choice at signup. Paid plan bills past credits. |

Sources: [Oracle Always Free Resources](https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [Google Cloud Free Program](https://docs.cloud.google.com/free/docs/free-cloud-features), [AWS Free Tier](https://aws.amazon.com/free/).

No free GPU suitable for uninterrupted BiLSTM or FinBERT training showed up as an always-on inclusion. Treat GPU time as a separate, explicitly approved session with checkpoints.

## 3. CPU, RAM, and storage requirements

| Profile | CPU | RAM | Disk | What runs |
|---|---|---|---|---|
| Laptop client | anything that runs a browser | — | git checkout | UI via SSH tunnel or local `gig serve` for development |
| Always-on paper / terminal | 2 ARM OCPU | 4 GB comfortable, 12 GB on the recommended shape | 40 GB plus the lake | `gig serve`, daily Yahoo ingest of the traded universe, reconciliation |
| Heavy research | more than the free VM | 16 GB+ | tens of GB for model caches | walk-forward ranker on a wide universe, later NLP batches |

The diagnostic route loads up to 40 names and refits a 2-state HMM and a GARCH(1,1). That is a research request, not a tick loop. Do not put it on a timer on the 1 GB `e2-micro`.

## 4. Expected workload capacity

On 2 OCPU / 12 GB: one uvicorn worker, a factor snapshot of a few hundred names, and a manual model refresh. A full-universe Yahoo ingest is network-bound and can take the better part of an hour the first time; it should run when the terminal is idle.

Not in capacity: tick microstructure, a second copy of `live_data_server.py`, or concurrent hyperparameter search.

## 5. Models that can run on CPU

Factors, IC weights, factor risk, optimizer, historical VaR/ES, sklearn or LightGBM ranker inference, lexicon sentiment, Gaussian HMM, GARCH(1,1), EWMA, Kalman, Black-Litterman, Kelly, exposure scale, Almgren-Chriss planner, Black-Scholes.

## 6. Workloads that want an optional GPU

FinBERT batch scoring, BiLSTM training, large embedding indexes. None of these are on the order path. If a GPU session dies, training must resume from a checkpoint. This repository does not yet write those checkpoints because those trainers are not in `src/gig`.

## 7. Estimated resource usage

Not a benchmark from this change. Order-of-magnitude from the code that exists:

- `gig serve` idle: well under 1 GB.
- Factor cloud on ~200 names: a few hundred MB and a couple of seconds.
- `/api/models/market` on ~40 names × ~800 sessions: seconds, not minutes, on a desktop core. Re-measure on the A1 before scheduling it.
- Synthetic backtest of the flagship sleeve: seconds to a few minutes depending on universe size.

No p50/p95 figures are claimed.

## 8. Free-tier limitations

- Oracle capacity is first-come in the home region. Signup can succeed and instance creation can still fail.
- Always Free is not a support contract. Reboots and host maintenance happen.
- Google egress of 1 GB/month will not carry a chatty public UI.
- AWS credits expire.
- ARM versus x86: install wheels for the platform you deploy. `gig._speed` must be compiled on that machine (`python3-dev`, a compiler). If the build fails, neutralization falls back to Python.

## 9. Billing and overage risks

- Oracle: a shape above the always-free OCPU or memory envelope, a second region, or a paid database.
- Google: any GPU, a larger machine type, disk beyond 30 GB-months, egress beyond 1 GB, a region outside the three US regions.
- AWS: converting the Free plan to Paid, or leaving an instance running after credits end.

Do not attach a paid upgrade from this repository's scripts. There are no scripts that call a cloud API.

## 10. Deployment commands

On the VM, after a git clone and a venv:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv build-essential
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev,web]"
.venv/bin/python -m gig doctor
.venv/bin/python -m gig db init
.venv/bin/python -m gig serve --host 127.0.0.1 --port 8000
```

Copy `.env` onto the VM only for keys you intend to use. Do not commit it.

From the laptop:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<vm>
# browser: http://127.0.0.1:8000
```

## 11. Remote development commands

```bash
ssh <user>@<vm>
cd ~/GIG_Quant_Terminal
git pull
.venv/bin/python -m pytest -q --tb=no
.venv/bin/python -m gig backtest --source synthetic
```

Heavy Yahoo backtests stay on the VM. The laptop does not need to compile the C++ extension unless you want a local `gig._speed`.

## 12. Backup and recovery

```bash
# on the VM, service stopped
cp data/gig.duckdb data/gig.duckdb.$(date +%Y%m%d)
```

Restore by stopping the server, replacing the file, and starting again. Then run `gig trade status` before `gig trade run`. The account, not the file, is the position source of truth once a broker is configured.

## 13. Paper-trading restart

1. Start `gig serve` only after DuckDB opens.
2. `gig trade status` to read positions from Alpaca.
3. `gig trade plan` and read the gate. It sends nothing.
4. Submit only with `gig trade run --yes` when the plan is the one you approved.
5. If the lake is stale or the broker errors, do not retry in a loop. The pre-trade gate is there to stop that.
6. `gig trade flatten --yes` is the kill switch for an open paper book.

This change did not execute steps 2–6. No Alpaca key was available, and connectivity is not claimed.

## 14. Upgrade path

If the free VM cannot finish a daily ingest and still serve the terminal:

- First drop the universe and turn off `/api/models/market` on a timer.
- Then move research to a second machine you explicitly pay for.
- Do not scale the Oracle shape past 2 OCPU / 12 GB on an Always Free tenancy unless you have accepted the overage.

Live trading is not an upgrade step. It stays off.
