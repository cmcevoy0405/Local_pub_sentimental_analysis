import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#Load dataset in, indexing first column
df = pd.read_csv('reviews.csv', index_col = 0)

#Remove time from date
df['Date'] = pd.to_datetime(df['Date']).dt.date

#Run on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Select model
model_id = "cardiffnlp/twitter-roberta-base-sentiment" # @param {type:"string"}
select_model = 'cardiffnlp/twitter-roberta-base-sentiment' # @param ["cardiffnlp/twitter-roberta-base-sentiment", "lxyuan/distilbert-base-multilingual-cased-sentiments-student"]
model_id = select_model
print("Selected Model: ", model_id)

#Load in pre trained model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels = 3
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

#Fit a pipeline for sentiment analysis
nlp = pipeline(
    'sentiment-analysis',
    model = model,
    tokenizer = tokenizer,
    device = device
)

#Labels Dictionary
labels = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

#Test sample
sample_text = df['Review'][57]
results = nlp(sample_text)
print(results)

#Access retriever for embedding
from sentence_transformers import SentenceTransformer

retriever = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device = device
)

import lancedb

db = lancedb.connect("./.lancedb")

#Function to access label scores and sentiment
def get_sentiment(reviews):
    sentiments = nlp(reviews)
    l = [labels[x["label"]] for x in sentiments]
    s = [x["score"] for x in sentiments]
    return l, s

import datetime

import datetime

def get_timestamp(dates):
    # Create a list to hold the timestamps
    timestamps = []
    for date in dates:
        # Check if the date is a datetime or date object
        if isinstance(date, datetime.datetime):
            # Append the timestamp for the datetime object
            timestamps.append(date.timestamp())
        elif isinstance(date, datetime.date):
            # Convert datetime.date to datetime.datetime and append the timestamp
            timestamps.append(datetime.datetime.combine(date, datetime.datetime.min.time()).timestamp())
        else:
            # Raise an error if the type is not recognized
            raise ValueError(f"Expected a date or datetime object, got {type(date)}.")

    return timestamps



from tqdm.auto import tqdm

batch_size = 32
data = []
for i in tqdm(range(0, len(df), batch_size)):
    # End of batch
    i_end = min(i + batch_size, len(df))
    # Extract batch
    batch = df.loc[i:i_end]
    # Generate embeddings per batch
    emb = retriever.encode(batch["Review"].tolist())
    # Convert to timestamp
    timestamp = get_timestamp(batch["Date"].tolist())
    batch["timestamp"] = timestamp
    # Sentiment per batch
    label, score = get_sentiment(batch["Review"].tolist())
    batch["label"] = label
    batch["score"] = score
    # Get metadata
    meta = batch.to_dict(orient="records")
    # Create unique ids
    ids = [f"{idx}" for idx in range(i, i_end)]
    # Add to upsert list
    to_insert = list(zip(ids, emb, meta))
    for id, emb, meta in to_insert:
        temp = {}
        temp['vector'] = emb
        for k, v in meta.items():
            temp[k] = v
        data.append(temp)


#create and insert records into lancedb table
tbl = db.create_table("tbl", data, mode = "overwrite")

def count_sentiment(result):
    sentiments = {
        "negative" : 0,
        "neutral" : 0,
        "positive": 0
    }

    for r in result:
        sentiments[r["label"]] += 1
    return sentiments

metadata = ["label", "Review", "Date", "timestamp"]

#Write queries for lancdb
queries = [
    "Was the food tasty?",
    "Was the food and drink reasonably priced?",
    "Were customers satisfied by the service from the staff?",
    "Is the location and atmosphere nice?",
    "How was management at the venue?"
]

import seaborn as sns
import matplotlib.pyplot as plt

num_queries = len(queries)

# Initialize subplots for each query
fig, axs = plt.subplots(nrows=1, ncols=num_queries, figsize=(40, 10))

# Loop through each query
for i, query in enumerate(queries):
    # Encode the query and search in the lancedb
    xq = retriever.encode(query).tolist()
    result = tbl.search(xq).select(metadata).limit(75).to_list()

    sentiment = count_sentiment(result)
    df = pd.DataFrame(list(sentiment.items()), columns=['Sentiment', 'Count'])

    # Plot the bar chart for current query on the subplot
    sns.barplot(x="Sentiment", y="Count", data=df, ax=axs[i], palette="Set2", edgecolor='black')
    axs[i].set_title(f"Query: {query}", fontsize=16, fontweight='bold')
    axs[i].set_xlabel("Sentiment", fontsize=14)
    axs[i].set_ylabel("Count", fontsize=14)
    for index, row in df.iterrows():
        axs[i].text(row['Sentiment'], row['Count'], int(row['Count']),
        color='black', ha='center', va='bottom', fontsize=12)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('sentiment_analysis_results.png', dpi=300)

plt.clf()
