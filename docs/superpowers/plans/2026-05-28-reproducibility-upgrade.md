# NeuralToneTransform Reproducibility Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current local-only NAM baseline success into a repo state that another machine can clone and reproduce with documented setup, path-safe configs, persisted logs, and an explicit verification script.

**Architecture:** Keep the current Step 1 and Step 2 workflow, but make every path derive from the repository root, make logs first-class artifacts, and add a reproducibility gate that checks environment, configs, artifacts, and logs. Python scripts own path-safe config generation and environment logging; PowerShell wrappers own tee-based logging and fail-fast orchestration on Windows.

**Tech Stack:** Python, PowerShell, PyTorch, Torchaudio, Librosa, Matplotlib, SoundFile, neural-amp-modeler

---

### Task 1: Dependency and Documentation Baseline

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Modify: `.gitignore`
- Modify: `reports/STEP1_STEP2_COMPLETION_REPORT.md`

- [ ] **Step 1: Capture the actual installed dependency versions**
- [ ] **Step 2: Write `requirements.txt` with installable package pins**
- [ ] **Step 3: Document the clone-to-run workflow in `README.md`**
- [ ] **Step 4: Tighten `.gitignore` without removing proof artifacts**
- [ ] **Step 5: Update the completion report with the reproducibility upgrade section**

### Task 2: Path-Safe Config Generation and Step 1 Logging

**Files:**
- Modify: `scripts/prepare_nam_baseline.py`
- Modify: `scripts/check_env.py`

- [ ] **Step 1: Confirm the current config is not reproducible**
- [ ] **Step 2: Refactor `scripts/prepare_nam_baseline.py` to derive paths from repo root**
- [ ] **Step 3: Make `scripts/check_env.py` tee its output into `logs/step1_env_check.log`**
- [ ] **Step 4: Verify the new config generation**

### Task 3: Step 2 Logging, Finalization Logging, and Reproducibility Gate

**Files:**
- Modify: `scripts/run_step2_nam_baseline.ps1`
- Modify: `scripts/finalize_step2_without_rerun.ps1`
- Create: `scripts/verify_reproducibility.ps1`
- Create: `logs/.gitkeep`

- [ ] **Step 1: Add tee-based prepare/training logging to `scripts/run_step2_nam_baseline.ps1`**
- [ ] **Step 2: Add tee-based archive logging to `scripts/finalize_step2_without_rerun.ps1`**
- [ ] **Step 3: Add `scripts/verify_reproducibility.ps1`**
- [ ] **Step 4: Run the fresh verification sequence**
