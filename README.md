# Quick Start


## Step 1: Download the MMLU Data

To download the required MMLU dataset, run:

```bash
python data_prepare.py
```

Your downloaded data directory should be structured as follows:

```
mmlu_data
├── mmlu_auxiliary_train.jsonl
├── mmlu_dev.jsonl
├── mmlu_test.jsonl
└── mmlu_validation.jsonl
```

---

## Step 2: Rephrase Task

Install necessary dependencies:

```bash
pip install -r requirements1.txt
```

Run the rephrasing script:

```bash
python rephrase_mmlu.py
```

---

## Step 3: Scoring Task

Install dependencies for scoring in another environment:

```bash
pip install -r requirements2.txt
```

Run the scoring script:

```bash
python score_mmlu.py
```

---

## Directory Summary

```
.
├── data_prepare.py
├── rephrase_mmlu.py
├── score_mmlu.py
├── requirements1.txt
├── requirements2.txt
└── mmlu_data/
```

---

## Notes

- Ensure Python and pip are properly installed and configured on your system.
- Check for any potential errors after running each script to confirm correct execution.





