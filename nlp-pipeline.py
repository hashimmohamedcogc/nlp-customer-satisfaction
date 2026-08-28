# Customer Satisfaction Analysis: Tesco vs Sainsburys
# NLP and Machine Learning Pipeline
#
# Author: Hashim Mohamed
# Degree: BSc (Hons) Business Technology, First Class Honours
# University: University of the West of Scotland
# Submitted: April 2026
#
# This script runs the complete eight-stage NLP pipeline used in my
# final year honours dissertation. It loads Trustpilot review data,
# cleans it, runs VADER sentiment analysis, trains a Naive Bayes
# classifier, and produces keyword frequency analysis and charts.

import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from wordcloud import WordCloud


# STAGE 1: LOAD THE DATA
# -------------------------------------------------------
# The dataset is a CSV file containing Trustpilot reviews
# for Tesco and Sainsburys collected between November 2025
# and February 2026. Update the file path if needed.

print("Stage 1: Loading data...")

df = pd.read_csv("data/sample-reviews.csv")

print("Number of reviews loaded:", len(df))
print("Columns:", list(df.columns))
print(df.head())


# STAGE 2: PRE-PROCESSING
# -------------------------------------------------------
# Before running sentiment analysis the text needs to be
# cleaned. Missing values are filled with empty strings,
# the review title and body are merged into one field,
# and the clean_text function removes noise from the text.

print("\nStage 2: Pre-processing the text...")

df["review_title"]   = df["review_title"].fillna("")
df["review_content"] = df["review_content"].fillna("")

df["full_review"] = df["review_title"] + " " + df["review_content"]


def clean_text(val):
    # Convert to string in case of any non-string values
    val = str(val)
    # Convert all text to lowercase
    val = val.lower()
    # Remove anything that is not a letter or a space
    val = re.sub(r"[^a-zA-Z\s]", " ", val)
    # Collapse multiple spaces into one
    val = re.sub(r"\s+", " ", val)
    # Remove leading and trailing whitespace
    val = val.strip()
    return val


df["clean_review"] = df["full_review"].apply(clean_text)

print("Pre-processing done. Sample output:")
print(df[["company_name", "clean_review"]].head())


# STAGE 3: VADER SENTIMENT ANALYSIS
# -------------------------------------------------------
# VADER (Valence Aware Dictionary and sEntiment Reasoner)
# is applied to every review. It produces a compound score
# between -1.0 and +1.0. The classify_sentiment function
# applies the standard VADER thresholds to assign a label.

print("\nStage 3: Running VADER sentiment analysis...")

analyzer = SentimentIntensityAnalyzer()

df["compound_score"] = df["clean_review"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)


def classify_sentiment(score):
    # Scores above 0.05 are Positive
    # Scores below -0.05 are Negative
    # Everything else is Neutral
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["compound_score"].apply(classify_sentiment)

print("Sentiment labels assigned. Sample:")
print(df[["company_name", "compound_score", "sentiment"]].head())


# STAGE 4: SENTIMENT SUMMARY TABLES
# -------------------------------------------------------
# Group by retailer and sentiment to produce the summary
# tables showing absolute counts and percentage distributions.

print("\nStage 4: Building sentiment summary tables...")

summary = df.groupby(["company_name", "sentiment"]).size().unstack(fill_value=0)
print("\nAbsolute counts:")
print(summary)

summary_pct = summary.div(summary.sum(axis=1), axis=0).mul(100).round(1)
print("\nPercentage distribution:")
print(summary_pct)

summary.to_excel("sentiment_counts.xlsx")
summary_pct.to_excel("sentiment_percentages.xlsx")
print("Tables saved to Excel.")


# STAGE 5: MACHINE LEARNING CLASSIFIER
# -------------------------------------------------------
# The cleaned review text is vectorised using TF-IDF and
# a Multinomial Naive Bayes classifier is trained on 80
# percent of the data. The remaining 20 percent is used
# as the test set. random_state=42 ensures the results
# are reproducible.

print("\nStage 5: Training Multinomial Naive Bayes classifier...")

X = df["clean_review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(stop_words="english")
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print("Overall Accuracy:", round(accuracy * 100, 1), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

report_dict = classification_report(y_test, predictions, output_dict=True)
pd.DataFrame(report_dict).transpose().round(2).to_excel("classification_report.xlsx")
print("Classification report saved to Excel.")


# STAGE 6: CONFUSION MATRIX
# -------------------------------------------------------
# The confusion matrix shows exactly how the model
# predicted each sentiment category. This revealed that
# the model failed entirely on neutral reviews due to
# class imbalance. Only 95 of the 2000 reviews were
# neutral which was not enough for the classifier to
# learn that class properly.

print("\nStage 6: Generating confusion matrix...")

labels = ["Negative", "Neutral", "Positive"]
cm     = confusion_matrix(y_test, predictions, labels=labels)
disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Sentiment Classification")
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=300)
plt.show()
print("Saved: charts/confusion_matrix.png")


# STAGE 7: WORD CLOUDS
# -------------------------------------------------------
# Word clouds are generated for all reviews combined,
# Tesco reviews only and Sainsburys reviews only.

print("\nStage 7: Generating word clouds...")


def make_wordcloud(text, title, filename):
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100)
    wc.generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("charts/" + filename, dpi=300)
    plt.show()
    print("Saved: charts/" + filename)


all_text       = " ".join(df["clean_review"])
tesco_text     = " ".join(df[df["company_name"] == "Tesco"]["clean_review"])
sainsbury_text = " ".join(df[df["company_name"] == "Sainsbury's"]["clean_review"])

make_wordcloud(all_text,       "Most Frequent Words in All Reviews",          "wordcloud_all.png")
make_wordcloud(tesco_text,     "Most Frequent Words in Tesco Reviews",        "wordcloud_tesco.png")
make_wordcloud(sainsbury_text, "Most Frequent Words in Sainsburys Reviews",   "wordcloud_sainsburys.png")


# STAGE 8: KEYWORD FREQUENCY ANALYSIS
# -------------------------------------------------------
# The top 15 keywords are extracted from Sainsburys reviews
# and from all negative reviews combined. A custom stop word
# list removes the store names and common filler words that
# would otherwise dominate the results.

print("\nStage 8: Keyword frequency analysis...")

base_stops  = set(stopwords.words("english"))
extra_stops = {"sainsbury", "sainsburys", "tesco", "always", "time", "use"}
all_stops   = base_stops.union(extra_stops)


def get_top_keywords(text, n=15):
    words    = text.split()
    filtered = [w for w in words if w not in all_stops and len(w) > 3]
    return Counter(filtered).most_common(n)


top_sainsburys = get_top_keywords(sainsbury_text)
top_negative   = get_top_keywords(" ".join(df[df["sentiment"] == "Negative"]["clean_review"]))

print("\nTop 15 Keywords in Sainsburys Reviews:")
for word, count in top_sainsburys:
    print(" ", word, ":", count)

print("\nTop 15 Keywords in Negative Reviews:")
for word, count in top_negative:
    print(" ", word, ":", count)

pd.DataFrame(top_sainsburys, columns=["Keyword", "Frequency"]).to_excel("sainsburys_keywords.xlsx", index=False)
pd.DataFrame(top_negative,   columns=["Keyword", "Frequency"]).to_excel("negative_keywords.xlsx",   index=False)

words  = [item[0] for item in top_negative]
counts = [item[1] for item in top_negative]

plt.figure(figsize=(10, 5))
plt.bar(words, counts, color="#2563EB")
plt.title("Top 15 Complaint Keywords in Negative Reviews")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("charts/top_complaint_keywords.png", dpi=300)
plt.show()
print("Saved: charts/top_complaint_keywords.png")

plot_data = summary[["Negative", "Neutral", "Positive"]]
plot_data.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=["#EF4444", "#F59E0B", "#10B981"]
)
plt.title("Sentiment Comparison: Tesco vs Sainsburys")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("charts/sentiment_comparison.png", dpi=300)
plt.show()
print("Saved: charts/sentiment_comparison.png")

print("\nPipeline complete. All outputs saved.")
