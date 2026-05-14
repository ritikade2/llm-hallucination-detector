# LLM Hallucination Detector

A lightweight, statistical framework for detecting hallucinations in LLM outputs. No ML models. No fine-tuning. Just Python and pandas.

Tested on 10,000 examples from the [HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) dataset.

## Results

| Approach | Precision | Recall |
|----------|-----------|--------|
| Soft Flag (score >= 1) | 0.71 | 0.96 |
| Strict Flag (score >= 3) | 1.00 | 0.38 |

## How It Works

Four statistical signals combine into a single hallucination score (0 to 4):

1. **Length Ratio** — hallucinated answers are longer relative to source knowledge
2. **Unknown Word Rate** — hallucinated answers introduce words not in the source
3. **Question-Answer Overlap** — hallucinated answers repeat question words back
4. **Numeric Inconsistency** — hallucinated answers introduce numbers not in the source

## Setup

```bash
git clone https://github.com/ritikade2/llm-hallucination-detector.git
cd llm-hallucination-detector
python -m venv .venv
source .venv/bin/activate
pip install datasets pandas matplotlib scipy
```

## Run

```bash
python download_data.py
python detect_hallucination.py
python visualize_signals.py
```

## Article

Full writeup on Towards Data Science: [link coming soon]

## License

MIT
