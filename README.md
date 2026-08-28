# Customer Satisfaction Analysis: Tesco vs Sainsbury's

NLP and machine learning pipeline built for my final year honours dissertation, BSc (Hons) Business Technology, University of the West of Scotland. Awarded First Class Honours, April 2026.

I analysed 2,000 Trustpilot reviews (1,000 each for Tesco and Sainsbury's) to compare customer sentiment between the two retailers, then mapped the findings onto the SERVQUAL service quality framework to turn them into something a retail business could actually act on. This was the first data project I built end to end in Python, and it's the main technical piece behind my move into data analysis.

## What it does

The pipeline runs in eight stages:

1. Loads the raw reviews and checks the data is complete
2. Cleans the text — merges title and body, lowercases, strips punctuation, collapses whitespace
3. Runs VADER sentiment analysis on every review and assigns Positive / Negative / Neutral
4. Builds summary tables of sentiment by retailer
5. Trains a Multinomial Naive Bayes classifier on TF-IDF vectors (80/20 train/test split)
6. Produces a confusion matrix to see where the model actually fails
7. Generates word clouds for Tesco, Sainsbury's and the combined set
8. Extracts the top complaint keywords from negative reviews and charts them

## Results

Sainsbury's came out well ahead — 68.1% positive reviews against Tesco's 41.6%. Tesco's negative rate (54.3%) means more than half of everything written about them was negative, which is a real commercial problem given how strongly review sentiment tracks purchase intent.

The complaints cluster around delivery, service, staff and order handling — mostly Reliability and Responsiveness in SERVQUAL terms. Sainsbury's positive reviews lean on staff helpfulness and the Nectar loyalty scheme, so some of that gap could be reciprocity rather than pure service quality.

The classifier hit 68.5% accuracy with a weighted F1 of 0.65, which is in line with published benchmarks for this kind of three-class sentiment task. It completely failed on the Neutral class though — only 95 of the 2,000 reviews were neutral (4.75%), nowhere near enough for the model to learn that category. Worth knowing before trusting a classifier like this on similarly imbalanced data.

## Stack

Python 3, pandas, scikit-learn, NLTK, vaderSentiment, matplotlib, wordcloud, Google Colab, Excel.

## Repo layout

```
nlp-pipeline.py          full pipeline, all eight stages, commented
requirements.txt         dependencies
data/sample-reviews.csv  20-row sample showing the expected format
charts/                  output folder — populated when you run the script
portfolio-summary.pdf    write-up with the full results and academic context
full-dissertation-report.docx   complete honours dissertation, 10,000+ words, full methodology, literature review and references
```

The full 2,000-row dataset isn't included since it wasn't originally mine to publish in full — the sample shows the exact structure the script expects.

## Running it

```
git clone https://github.com/hashimmohamedcogc/nlp-customer-satisfaction.git
cd nlp-customer-satisfaction
pip install -r requirements.txt
```

Drop your own review data into `data/`, matching the column layout in the sample CSV, and point `nlp-pipeline.py` at it (line 44). Then:

```
python nlp-pipeline.py
```

Charts land in `charts/`, and the summary tables and keyword frequencies get written out to Excel automatically.

## Academic context

- Module: COMP10034 Computing Honours Project
- Frameworks used: SERVQUAL (Parasuraman et al., 1988), eWOM Theory (Donthu et al., 2021)
- 48 peer-reviewed sources, 2009–2026

## About the author

BSc (Hons) Business Technology, First Class Honours, University of the West of Scotland, 2026. More projects on my [profile](https://github.com/hashimmohamedcogc), or reach me on [LinkedIn](https://linkedin.com/in/hashimmohamedcogc).
