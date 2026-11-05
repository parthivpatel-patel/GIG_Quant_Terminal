# Market flow feasibility

Reviewed 2026-11-05 from pages this environment could retrieve. No options-sweep, dark-pool print, or institutional order-flow feed is connected. No FINRA or SEC transaction file was downloaded.

## SEC insider transaction disclosures

The Data.gov catalog entry [Insider Transactions Data Sets](https://catalog.data.gov/dataset/insider-transactions-data-sets) returned HTTP 200. Its JSON-LD `dateModified` is `2026-07-09T11:17:07Z`. The description says the sets are extracted from the XML portions of Forms 3, 4, and 5, presented without change from the as-filed submissions, and that the Commission does not guarantee accuracy. It says the data is not a substitute for the filings.

The catalog lists 82 quarterly zip resources, including `2023q1` through `2023q4` and a latest file named `2026q2_form345.zip`. The JSON-LD `license` is `https://www.usa.gov/government-copyright`. That page also returned HTTP 200. It says federal website materials may be copyrighted or otherwise restricted, and that a user should check with the agency before using them. It is not a public-domain grant.

`https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2023q1_form345.zip` and `https://www.sec.gov/files/insider_transactions_readme.pdf` both returned HTTP 403 from this environment. The SEC data-research pages returned HTTP 403 as well. No insider file was parsed, and no insider study was run.

EDGAR itself stays unwired. `docs/integrations/API_REQUIREMENTS.md` already records that a real contact email is required in `EDGAR_USER_AGENT` before any EDGAR request. This pass did not ask for one and did not call EDGAR.

## FINRA aggregate off-exchange statistics

`https://www.finra.org/finra-data/otc-transparency` and `https://www.finra.org/finra-data` returned HTTP 403. `https://otctransparency.finra.org/` returned HTTP 200. The document title is "OTC Transparency Data". The visible footer says the mark is reserved. The application bundle `main-BJMUYDGJ.js` (retrieved with that page, HTTP 200) contains the terms a visitor must accept before the data is shown:

> All the information on this website is published in good faith and for general information purpose only and subject to change without notice. The information is based on data reported to FINRA, and FINRA does not make any warranties about the completeness, timeliness, reliability or accuracy of this information. ... Attribution shall be given to FINRA in all copyrightable works which are created using the information contained on this website. ... OTC Transparency data is provided via http://www.finra.org/industry/OTC-Transparency and is copyrighted by FINRA.

The same bundle says access requires agreeing to those terms. No FINRA file was downloaded, vendored, or drawn. Aggregate off-exchange statistics are not in the terminal.

## Mathematical options pricing

`gig.options.black_scholes` already prices a European contract and returns Greeks labelled `MODEL_ESTIMATE`. Those numbers are a formula. They are not observed option trades, sweeps, or open interest. The existing options workspace uses that kernel on admitted or authored inputs. This pass did not add a second pricer and did not present the formula as flow.

## What was not built

A Market Flow screen needs a file this process can read under terms that allow the use. The SEC zips were blocked, and the FINRA application is copyrighted and gated. Building a chart from either source now would mean inventing the observations. No market-flow route or panel was added.

The Treasury par-yield study in the research organization is a yield-curve question. It is not options flow and not off-exchange volume.
