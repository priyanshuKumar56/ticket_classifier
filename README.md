# Clinic Support Ticket Classifier

A small, local, production-minded **educational** project. It sorts short, **fictional** clinic-software support tickets into four categories using TF-IDF features and Logistic Regression, and ships with a Streamlit interface, a command-line toolkit and an automated test suite.

> **Educational prototype - not a clinical decision-making tool.** It classifies fictional *software support tickets*. It must never be used for patient care, patient triage or any medical decision, and you should never enter real patient information.

This initial version deliberately contains **four known, documented bugs** for a debugging exercise. See [Educational debugging exercises](#educational-debugging-exercises).

---

## Contents

1. [What problem it solves](#what-problem-it-solves)
2. [What is in the box](#what-is-in-the-box)
3. [Setup (macOS, Linux, Windows)](#setup-macos-linux-windows)
4. [Daily commands](#daily-commands)
5. [Architecture and file responsibilities](#architecture-and-file-responsibilities)
6. [Data: schema, labels and demo data](#data-schema-labels-and-demo-data)
7. [How the machine-learning pipeline works](#how-the-machine-learning-pipeline-works)
8. [Interpreting metrics and the confusion matrix](#interpreting-metrics-and-the-confusion-matrix)
9. [Limitations and responsible use](#limitations-and-responsible-use)
10. [Educational debugging exercises](#educational-debugging-exercises)
11. [Troubleshooting](#troubleshooting)
12. [Useful links](#useful-links)

---

## What problem it solves

A clinic-software support team receives many free-text messages ("I can't reset my password", "the invoice total looks wrong"). Sorting them by hand is slow. This project shows, end to end, how to build a transparent text classifier that proposes a category, shows *why*, and flags uncertain tickets for a person to review.

It is a teaching project. It demonstrates text preprocessing, TF-IDF, Logistic Regression, leakage-free evaluation, probability estimates, error handling and testing - not a deployable triage system.

## What is in the box

- **Classifier**: `TfidfVectorizer` + `LogisticRegression` in one scikit-learn `Pipeline` (vectoriser fitted on the training split only).
- **Streamlit app** (light theme): classify a ticket, per-category probability estimates, "why this prediction?" term chart, sample selector, batch classification of a CSV, evaluation dashboard with confusion matrix, misclassification review, and a permanent educational-prototype warning.
- **Human-review flag**: tickets with a low top probability, a small margin between the top two categories, or no known vocabulary are marked "needs review".
- **Explanations**: the words that pushed the score for the predicted category up or down.
- **Honest evaluation**: stratified split, cross-validation on the training split only, optional hyper-parameter search (`--tune`) that never touches the test set, a "always guess the most common class" baseline for context.
- **Safe-ish persistence**: the saved model's SHA-256 checksum is stored in a metadata file and verified before loading (see [Limitations](#limitations-and-responsible-use) for what this does and does not protect against).
- **Re-evaluation script** that reloads the model and scores it on the *exact* test tickets recorded at training time, refusing to run if the dataset changed.
- **Tests**: pytest suite with temporary files and small fixtures, no network access; headless Streamlit UI tests; four strict expected-failure tests for the planted bugs.
- **Demo data**: 120 labelled synthetic tickets plus a 24-ticket unlabelled demo batch.
- **CI file** for GitHub Actions (`.github/workflows/ci.yml`). It is optional and was not executed while building this project.

All data is original, hand-written and fictional. No real patient data, external datasets, paid APIs, external LLMs, databases or cloud services are used.

---

## Setup (macOS, Linux, Windows)

### 1. Install the prerequisites

| Tool | Why | Download / docs |
|---|---|---|
| Python 3.11 or newer (3.12 was used for testing) | runs everything | <https://www.python.org/downloads/> |
| pip | installs packages (ships with Python) | <https://pip.pypa.io/en/stable/installation/> |
| A terminal | macOS: Terminal; Linux: any shell; Windows: PowerShell | - |
| Git (optional) | version control | <https://git-scm.com/downloads> |
| VS Code (optional) | editor with a Python extension | <https://code.visualstudio.com/download> |

Check your Python version:

```bash
# macOS / Linux
python3 --version
```

```powershell
# Windows (PowerShell)
py -3 --version
```

> Windows tip: tick **"Add python.exe to PATH"** in the Python installer. Linux (Debian/Ubuntu) may also need `sudo apt install python3-venv`.

### 2. Open the project folder

Unzip the project, then move into it:

```bash
cd clinic-ticket-classifier
```

### 3. Create and activate a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer ([venv documentation](https://docs.python.org/3/tutorial/venv.html)).

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, allow it for this window only and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd.exe)**

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt should now start with `(.venv)`. Run all later commands from the project root with the environment active. Type `deactivate` to leave it.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Train, test and launch

```bash
python train.py            # train + evaluate + save to results/
python -m pytest           # run the test suite
streamlit run app.py       # open the interface (http://localhost:8501)
```

If Streamlit asks for an email address on first launch, just press Enter.

### Versions this project was tested with

Linux (Ubuntu), Python 3.12.3: numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, streamlit 1.66.1, pytest 9.1.1. `requirements.txt` states only lower bounds. macOS, Windows and Python 3.11 were **not** exercised while building this project; the CI workflow is there so you can check them.

---

## Daily commands

Run from the project root with the virtual environment active.

| Goal | Command |
|---|---|
| Train, evaluate once on the held-out test split, save everything | `python train.py` |
| Same, but choose hyper-parameters by cross-validation on the training split | `python train.py --tune` |
| Train into another folder | `python train.py --output-dir my_results` |
| Re-evaluate the saved model on its recorded test tickets | `python evaluate.py` |
| Classify one ticket | `python predict.py --text "The reminder emails are not being sent"` |
| ... with the words that drove the decision | `python predict.py --text "..." --explain` |
| ... as JSON | `python predict.py --text "..." --json` |
| Classify a CSV (needs a `text` column) | `python predict.py --file data/demo_batch.csv --output results/batch_predictions.csv` |
| Launch the interface | `streamlit run app.py` |
| Launch on another port | `streamlit run app.py --server.port 8600` |
| Run all tests | `python -m pytest` |
| Run one test file, verbosely | `python -m pytest tests/test_model.py -v` |

Use `--model-dir DIR` with `predict.py` / `evaluate.py` if you trained into another folder.

**Expected test summary for this initial version:** `81 passed, 4 xfailed`. The four expected failures are the planted bugs. Once you fix a bug, its strict `xfail` test will report `XPASS(strict)` as a failure until you remove the marker - that is your signal that the fix works.

**What `train.py` writes to `results/`** (git-ignored; recreate any time):

| File | Content |
|---|---|
| `model.joblib` | the fitted pipeline |
| `model_metadata.json` | seed, dataset fingerprint, test ticket IDs, hyper-parameters, library versions, model checksum |
| `evaluation.json` | metrics, classification report, confusion matrix, baseline, cross-validation, review-flag summary |
| `test_predictions.csv` | every test ticket with true/predicted category and probabilities |
| `misclassified.csv` | wrongly classified test tickets, most confidently wrong first |
| `confusion_matrix.png` | the confusion matrix as an image |

The Streamlit app trains automatically on first launch if `results/` has no model.

---

## Architecture and file responsibilities

```
clinic-ticket-classifier/
  app.py                  Streamlit interface (presentation only)
  train.py                CLI: train, evaluate, save
  predict.py              CLI: classify one ticket or a CSV
  evaluate.py             CLI: re-score the saved model on its recorded test split
  data/
    tickets.csv           120 labelled synthetic tickets
    demo_batch.csv        24 unlabelled demo tickets (sample selector and batch tab)
  src/
    config.py             all constants: labels, seed, limits, thresholds, file names
    preprocessing.py      text normalisation used by the TF-IDF step
    validation.py         ticket-text and dataset validation + error types
    data_loader.py        load/tidy CSV, stratified split, dataset fingerprint
    model.py              pipeline, training, cross-validation, tuning, prediction
    explain.py            per-prediction term contributions
    evaluation.py         metrics, confusion matrix, baseline, misclassification review
    artifacts.py          save/load the model with checksum verification
    training.py           end-to-end training run shared by train.py and app.py
  tests/                  pytest suite (+ conftest.py fixtures)
  results/.gitkeep        generated artifacts land here
  .streamlit/config.toml  light theme, usage statistics off
  .github/workflows/ci.yml  optional GitHub Actions test matrix
  pytest.ini  requirements.txt  .gitignore  README.md
```

Data flows one way: `data_loader` (validated data) -> `model` (fit/predict) -> `evaluation` (metrics) -> `artifacts` (save). The interface and the CLIs only call these modules; they contain no ML logic of their own.

## Data: schema, labels and demo data

`data/tickets.csv` columns:

| Column | Type | Rules |
|---|---|---|
| `ticket_id` | string | stable ID such as `T-001`; must be unique |
| `text` | string | one short fictional ticket; must not be blank |
| `category` | string | one of `appointment`, `billing`, `login`, `technical` |

Labels are matched case-insensitively and surrounding whitespace is trimmed on load. At least 6 examples per label are required.

### Label definitions

Label by the **primary thing the sender needs fixed**, not by which words appear.

| Label | Meaning | Examples |
|---|---|---|
| `appointment` | scheduling behaviour and settings: booking, rescheduling, cancelling, reminders, availability, slots, no-shows | "Patients receive two reminder messages for the same appointment." |
| `billing` | money: invoices, charges, refunds, fees, insurance claims, subscription price and plan | "A patient was charged a late cancellation fee even though they cancelled two days before." |
| `login` | getting into the system: passwords, verification codes, locked accounts, SSO, invitations, session expiry, access and permissions | "I cannot log in to see today's appointments, it says my session is invalid." |
| `technical` | software faults: crashes, error codes, slowness, blank screens, sync/integration failures, export and print problems | "The appointment confirmation page loads forever and then times out." |

Many tickets deliberately share vocabulary with other categories (note "appointment" in the login and technical examples above), so a keyword lookup would fail and word context matters. Some tickets are genuinely arguable; that is intentional and realistic.

### About the data

- 120 original, hand-written, **fictional** tickets (31 appointment, 30 billing, 28 login, 31 technical) with varied phrasing. The file order is shuffled with a fixed seed.
- It is a **small educational dataset**. It does not represent real clinic traffic, so its results cannot establish production readiness.
- `data/demo_batch.csv` (`ticket_id`, `text`; IDs `D-001`...) holds 24 *unlabelled* tickets that do not overlap the training data. It includes ambiguous tickets, an out-of-scope message and a gibberish line so you can see the review flag in action.
- To add data: append rows with new unique IDs, then run `python train.py` again. `evaluate.py` refuses to run against a dataset that changed after training.

## How the machine-learning pipeline works

1. **Load and validate** (`data_loader.py`, `validation.py`): required columns, no missing/blank values, only known labels, unique IDs, enough examples per label.
2. **Split**: a reproducible, label-stratified 75/25 split (`seed 42`). The test split is used for **one** final evaluation.
3. **Preprocess** (`preprocessing.py`): lower-case; mask URLs, e-mails and numbers; collapse whitespace.
4. **Fit** one `Pipeline`: `TfidfVectorizer` (1-2-word n-grams, sublinear TF) -> `LogisticRegression`. Because the vectoriser is a pipeline step, its vocabulary and IDF weights come only from the training split.
5. **Cross-validate** on the training split only (stratified folds; the pipeline is cloned and re-fitted per fold). `--tune` additionally grid-searches `ngram_range` and `C` the same way.
6. **Evaluate** once on the held-out test split: accuracy, per-class precision/recall/F1, macro averages, classification report, confusion matrix, misclassified tickets, plus a most-frequent-class baseline.
7. **Save** the pipeline and metadata atomically; verify the checksum on load.

Defaults (`ngram_range=(1, 2)`, `C=10`) were fixed in `src/config.py` before looking at any test result and are not adjusted against the test set.

**Probabilities are estimates.** They come from `predict_proba`. Regularised Logistic Regression trained on ~90 examples is not guaranteed to be calibrated, so "0.80" does not mean "right 80% of the time". The review thresholds in `config.py` are simple rules of thumb, not safety guarantees.

---

## Interpreting metrics and the confusion matrix

- **Accuracy** - share of test tickets labelled correctly. Compare it with the baseline shown next to it: with four roughly balanced categories, guessing the most common one already scores about 25-30%.
- **Precision (per category)** - of tickets *predicted* as this category, how many truly are. Low precision means false alarms.
- **Recall (per category)** - of tickets that truly *are* this category, how many were found. Low recall means missed tickets.
- **F1** - the balance of precision and recall for one category.
- **Macro F1** - F1 averaged over the four categories with **equal weight**, so a weak category cannot hide behind a big one. The classification report's `macro avg` row shows it; `weighted avg` weights categories by their number of test tickets.
- **Confusion matrix** - rows are the *true* category, columns the *predicted* one. The diagonal holds correct answers; every off-diagonal cell is a specific mix-up (for example "login tickets predicted as technical"). Read a row to see where one category's tickets went, a column to see what pollutes one prediction.
- **Sample size matters.** With about 30 test tickets, one ticket moves accuracy by roughly 3 points and a category's recall by 12-14 points. Treat every number as rough, and prefer the cross-validation line and the misclassification review for insight.

## Limitations and responsible use

- **Tiny, synthetic, tidy data.** Real tickets contain typos, abbreviations, other languages, attachments and context that these do not.
- **Results cannot establish production readiness**, and there is no calibration, drift monitoring, fairness analysis or security review.
- **Probabilities are not guaranteed confidence**; the review flag is a heuristic.
- **Explanations are model-specific**: the term chart shows linear-model weights, not causal reasons.
- **Model files are pickles.** `joblib.load` can execute code from a malicious file. The checksum catches corruption or a swapped model file, but not someone who can edit both the model and its metadata. Only load models you trained yourself; never load a `.joblib` from an untrusted source.
- **Not for clinical use**, patient triage, or any decision about a person's care. Do not enter real patient data.
- Local only: no database, authentication, cloud deployment or external LLM.

---

## Educational debugging exercises

The initial version contains **exactly four** small, deterministic defects. They are not random, do not corrupt data and do not make the app unusable; everything else works. Each has a **strict expected-failure test** (`@pytest.mark.xfail(strict=True)`) that states the *correct* behaviour. The defects are **not fixed** in this version.

### The four defects at a glance

| ID | Area | File | Function / location | Strict-xfail test |
|---|---|---|---|---|
| **A** | Input validation | `src/validation.py` | `validate_ticket_text` (empty-text check) | `tests/test_validation.py::test_whitespace_only_text_is_rejected_with_a_helpful_message` |
| **B** | Evaluation | `src/evaluation.py` | `compute_metrics` (the `macro_f1` line) | `tests/test_evaluation.py::test_macro_f1_uses_the_macro_average` |
| **C** | Interface state | `app.py` | `classify_tab` (the `except` branch of the Predict button) | `tests/test_app.py::test_error_clears_the_previous_result` |
| **D** | Data integrity | `src/validation.py` | `validate_dataset` (duplicate check) | `tests/test_validation.py::test_duplicate_ticket_ids_with_different_text_are_rejected` |

See all four expected failures with reasons:

```bash
python -m pytest -rx
```

See the *actual* failure of each (the bug) by ignoring the xfail markers:

```bash
python -m pytest --runxfail tests/test_validation.py tests/test_evaluation.py tests/test_app.py
```

### Bug A - whitespace-only ticket text is accepted

- **Expected behaviour:** whitespace-only text (spaces, tabs, newlines) is rejected with a helpful message such as "Ticket text is empty. Describe the problem in a sentence or two." The CLI exits with code 2; the app shows the error and no result.
- **Actual behaviour:** only the *empty string* is rejected. Whitespace-only text passes validation and receives a prediction (flagged for review because no words are known).
- **Reproduce (CLI):**

  ```bash
  python predict.py --text "   "
  ```

  Expected: `Error: Ticket text is empty...` and exit code 2. Actual: a predicted category is printed and the exit code is 0.
- **Reproduce (UI):** `streamlit run app.py`, type a few spaces into the ticket box, press **Predict Category**.
- **Reproduce (test):** `python -m pytest --runxfail tests/test_validation.py::test_whitespace_only_text_is_rejected_with_a_helpful_message` fails with `DID NOT RAISE InvalidTicketTextError`.
- **Note:** the batch path (`predict_dataframe`) goes through the same validator, so a whitespace-only row is also accepted there.

### Bug B - "macro F1" is really the weighted F1

- **Expected behaviour:** the reported `macro_f1` is the unweighted mean of the four per-category F1 scores (`f1_score(..., average="macro")`) and matches the `macro avg` row of the classification report.
- **Actual behaviour:** `compute_metrics` calculates it with the weighted average, so it matches the report's `weighted avg` row instead. It is wrong whenever categories have different numbers of test tickets, and wrong by a lot when classes are imbalanced.
- **Reproduce (deterministic snippet):**

  ```bash
  python -c "from src.evaluation import compute_metrics; t=['appointment']*6+['billing']*2+['login']*2+['technical']*2; p=['appointment']*6+['billing','appointment']+['login']*4; print(round(compute_metrics(t,p)['macro_f1'],4))"
  ```

  Expected `0.5641`. Actual `0.6838`.
- **Reproduce (real run):** `python train.py`, then compare the `macro F1` line with the `macro avg` row of the printed report (or the **Macro F1** card in the app's Evaluation tab with the per-category table). In the tested environment the card showed `0.5274` while the `macro avg` F1 was `0.5215`; the gap is small because the real test split is nearly balanced.
- **Reproduce (test):** `python -m pytest --runxfail tests/test_evaluation.py::test_macro_f1_uses_the_macro_average`.

### Bug C - stale result shown after an error

- **Expected behaviour:** when a prediction fails (for example empty input), the app shows the error and **no** result card, so nothing on screen can be mistaken for the answer to the latest input.
- **Actual behaviour:** the previous successful result stays on screen under the new error message, as if it belonged to the latest input.
- **Reproduce (UI):**
  1. `streamlit run app.py`, open the **Classify** tab.
  2. Enter `I forgot my password and the reset email never arrives` and press **Predict Category**. A result card appears.
  3. Delete all the text and press **Predict Category** again.
  4. The error "Ticket text is empty..." appears, **and** the old result is still shown below it.
- **Reproduce (headless):** `python -m pytest --runxfail tests/test_app.py::test_error_clears_the_previous_result` fails with `a stale result is still on screen after the error`.

### Bug D - duplicate ticket IDs are not detected

- **Expected behaviour:** `validate_dataset` raises `DatasetValidationError` ("Duplicate ticket_id value(s) found: ...") whenever any `ticket_id` appears more than once, whatever the other columns contain.
- **Actual behaviour:** a duplicate is detected only when the *whole row* is identical. Two different tickets sharing an ID pass validation, which would break anything that looks tickets up by ID (for example `evaluate.py`, which finds test tickets by ID).
- **Reproduce:**

  ```bash
  python -c "from src.data_loader import load_dataset; from src.validation import validate_dataset; df=load_dataset(); df.loc[1,'ticket_id']=df.loc[0,'ticket_id']; print('duplicates:', int(df['ticket_id'].duplicated().sum())); validate_dataset(df); print('accepted (should have raised)')"
  ```

  Expected: a `DatasetValidationError` traceback. Actual: prints `duplicates: 1` then `accepted (should have raised)`.
- **Reproduce (test):** `python -m pytest --runxfail tests/test_validation.py::test_duplicate_ticket_ids_with_different_text_are_rejected`.
- **Not a bug:** identical duplicate rows *are* rejected, and `data/tickets.csv` itself has unique IDs (checked by a normal test).

### After you fix a defect

1. Run its test with `--runxfail` and confirm it passes.
2. Run `python -m pytest`. The test now shows `XPASS(strict)` and fails the run on purpose - delete its `@pytest.mark.xfail(...)` decorator.
3. When all four are fixed, the whole suite should pass with no expected failures.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root (the folder containing `app.py`). |
| `ModuleNotFoundError: No module named 'streamlit'` (or sklearn, pandas...) | Activate the virtual environment, then `python -m pip install -r requirements.txt`. |
| `No trained model found ... Run python train.py first.` | Run `python train.py` (the app does this automatically on first launch). |
| `does not match the checksum recorded in model_metadata.json` | The model file changed or is corrupt. Delete `results/` contents and retrain. |
| Warning about a different scikit-learn version | Retrain with `python train.py` on the installed version. |
| `python` not found on Windows | Use `py -3` instead, or reinstall Python with "Add to PATH" ticked. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8600` |
| Streamlit asks for an email | Press Enter. |
| The app still shows an old model after retraining | Click **Reload model files** in the sidebar, or restart the app. |

## Useful links

Setup and tooling
- Python downloads: <https://www.python.org/downloads/>
- Virtual environments (venv): <https://docs.python.org/3/tutorial/venv.html>
- pip installation: <https://pip.pypa.io/en/stable/installation/>
- Git: <https://git-scm.com/downloads> - VS Code: <https://code.visualstudio.com/download>

Libraries used
- scikit-learn: <https://scikit-learn.org/stable/> - [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- pandas: <https://pandas.pydata.org/docs/>
- Streamlit: <https://docs.streamlit.io/> - [testing apps with AppTest](https://docs.streamlit.io/develop/concepts/app-testing)
- pytest: <https://docs.pytest.org/>
- joblib: <https://joblib.readthedocs.io/>
- matplotlib: <https://matplotlib.org/stable/>
