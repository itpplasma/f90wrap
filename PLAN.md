# Direct-C Risk Mitigation Plan (8 Oct 2025)

## Current Status Snapshot
- Branch: `feature/direct-c-clean`
- Direct-C harness (`python3 test_direct_c_compatibility.py`): **50 / 50 PASS (100 %)** at 08 Oct 2025 18:12 UTC.
- Helper/legacy unit suite (`pytest test/test_parser.py test/test_transform.py test/test_directc.py test/test_directc_e2e.py`): **37 / 37 PASS**.
- No tests or examples have been disabled; Direct-C and helper paths are both green after wrapper import fixes.

## Goals
1. Restore full pass status for the legacy/helper toolchain without regressing Direct-C coverage.
2. Minimise divergence from `origin/master` by scoping changes and reverting opportunistic diffs where possible.
3. Guarantee that no tests/examples are deactivated to achieve Direct-C parity; both modes must stay green.
4. Provide regular status reports (after every major test run and at least once per working session).

## Workstreams & Tasks

### 1. Helper Regression Remediation
- [x] Scope derived-type helper renaming to Direct-C generation only; keep legacy helper symbols unchanged.
- [x] Re-run targeted unit tests (`pytest test/test_transform.py`) to confirm binding counts align with historical expectations.
- [x] Restore default real/NumPy type mapping semantics (`numpy_utils`) to satisfy existing assertions without compromising Direct-C fixes.
- [x] Validate full helper unit suite (`pytest test/test_parser.py test/test_transform.py test/test_directc.py test/test_directc_e2e.py`).
- [ ] Document root causes and resolutions in commit notes / change log.

### 2. Direct-C Confidence Checks
- [x] Reconfirm `python3 test_direct_c_compatibility.py` after each remediation to ensure no regressions.
- [ ] Expand smoke tests (if practical) to cover newly scoped symbol names.

### 3. Diff-to-Master Audit & Reduction
- [ ] Produce a component-wise diff against `origin/master` focusing on high-churn files (`f90wrapgen.py`, `pywrapgen.py`, `directc_cgen.py`).
- [ ] Identify opportunities to isolate Direct-C behaviour through feature flags or localised modules rather than broad edits.
- [ ] Trim historical artefacts (debug scripts, stale reports) that are not required upstream.

### 4. Reporting Cadence
- Status updates here after: (a) each major pytest / harness run, (b) completion of any checklist item above, and (c) end-of-day summary.
- Capture test commands, pass/fail tallies, and outstanding blockers in every report.

## Deliverables
- Clean commit(s) with helper + Direct-C tests green.
- Updated documentation highlighting the scoped helper naming strategy.
- Summary note comparing the revised branch against `origin/master`, highlighting residual diffs and justification.
