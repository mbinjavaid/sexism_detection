import numpy as np
import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# # Parameters:
# Load pre-saved npy files (True) or compute encodings and cosine similarities again (False):
load = False
# Set the k and epsilon values in the formula: S - kN < epsilon (classification criterion):
k = 0.8
epsilon = 0.1065

# Directories for the datasets (should contain sexism_data.csv and sexism_annotations.csv):
gen_folder = "gpt"
dataset_folder = "Call me sexist but"

# # File names:
# File containing GPT-generated tweet-like scales:
gen_file = "gpt_tweets.csv"
# File containing 'Call me sexist but' dataset (binary labels):
data_file = "sexism_data.csv"
# File containing GPT-generated regular scales:
ask3_file = "ask3_sexist_tweetandscale.csv"

# The model to use for encoding the text: (can also try out different ones)
sbert_model = 'all-MiniLM-L6-v2'

gen_path = os.path.join(gen_folder, gen_file)
data_path = os.path.join(dataset_folder, data_file)
ask3_path = os.path.join(gen_folder, ask3_file)

gpt_tweets = pd.read_csv (gen_path)
data = pd.read_csv(data_path)
ask3 = pd.read_csv(ask3_path)

# Data pre-processing:

# Define selection criteria to extract scale items and tweets:
scales_mask_true = (gpt_tweets["sexist"] == True)
scales_mask_false = (gpt_tweets["sexist"] == False)
orig_scales_mask_true = (data["dataset"] == "scales") & (data["sexist"] == True)
orig_scales_mask_false = (data["dataset"] == "scales") & (data["sexist"] == False)

tweets_mask = (data["dataset"] != "scales")

# Get scale items (using all available, both GPT-generated and the original ones
# from 'Call me sexist but' dataset)
scales_true = gpt_tweets.loc[scales_mask_true, ["tweet"]]
scales_false = gpt_tweets.loc[scales_mask_false, ["tweet"]]
orig_scales_true = data.loc[orig_scales_mask_true, ["text"]]["text"].to_numpy()
orig_scales_false = data.loc[orig_scales_mask_false, ["text"]]["text"].to_numpy()
ask3_scales_true = ask3["out_scale"].to_numpy()

scales_true = scales_true["tweet"].to_numpy()
scales_true2 = ask3_scales_true
scales_true3 = orig_scales_true
# concatenate the two arrays:
scales_true = np.concatenate((scales_true, scales_true2, scales_true3))

for i in range(len(scales_true)):
    # Remove newline characters:
    scales_true[i] = scales_true[i].replace('\n', ' ')
    # Remove trailing whitespace:
    scales_true[i] = scales_true[i].strip()

scales_false = scales_false["tweet"].to_numpy()
scales_false2 = orig_scales_false
# concatenate the two arrays:
scales_false = np.concatenate((scales_false, scales_false2))

for i in range(len(scales_false)):
    # Remove newline characters:
    scales_false[i] = scales_false[i].replace('\n', ' ')
    # Remove trailing whitespace:
    scales_false[i] = scales_false[i].strip()

# Get (non-scale item) tweets and IDs:
tweets_ = data.loc[tweets_mask, ["id", "text", "sexist"]]
tweets_ = tweets_.drop_duplicates(subset=['id'], keep="first")
# tweets_ids = tweets_["id"].to_numpy()
tweets_true_labels = tweets_["sexist"].to_numpy()
tweets = tweets_["text"].to_numpy()

# Remove URLs, retweet tag at beginning, mention tag from tweets,
# since they do not really provide information about the text itself:
mention_tag = re.compile(r'\bMENTION\d+\b')
for i in range(len(tweets)):
    tweets[i] = re.sub(r'http\S*:{1}/{2}\S+', '', tweets[i]).strip()
    tweets[i] = tweets[i].removeprefix('RT')
    tweets[i] = re.sub(mention_tag, '', tweets[i]).strip()
    # Remove newline characters:
    tweets[i] = tweets[i].replace('\n', ' ')
    # Remove trailing whitespace:
    tweets[i] = tweets[i].strip()


# # Embedding the tweets and scale items:

# Define the model to use for embedding the text:
model = SentenceTransformer(sbert_model)

# Check if directory named 'scores' exists, if not, create it:
scores_folder = "scores"
if not os.path.exists(scores_folder):
    os.makedirs(scores_folder)

if not load:
    # Embed the scale items and tweets
    scales_true_embeddings = model.encode(scales_true, convert_to_tensor=True)
    scales_false_embeddings = model.encode(scales_false, convert_to_tensor=True)

    tweets_embeddings = model.encode(tweets, convert_to_tensor=True)

    # Cosine Similarity takes values in the range [-1,1]: -1 means opposite vectors,
    # 0 means uncorrelated vectors, 1 means similar vectors.
    cosine_scores_true = util.cos_sim(tweets_embeddings, scales_true_embeddings)
    cosine_scores_false = util.cos_sim(tweets_embeddings, scales_false_embeddings)

    # Compute the average similarity of each tweet to all sexist scale items:
    # (returns an array of length equal to the number of tweets)
    sexist_similarity = cosine_scores_true.mean(dim=1).detach().cpu().numpy()

    # Compute the average similarity of each tweet to all non-sexist scale items:
    # (returns an array of length equal to the number of tweets)
    non_sexist_similarity = cosine_scores_false.mean(dim=1).detach().cpu().numpy()

    cosine_scores_true = cosine_scores_true.cpu().detach().numpy()
    cosine_scores_false = cosine_scores_false.cpu().detach().numpy()

    np.save(scores_folder + "/cosine_scores_true_task3_merged", cosine_scores_true)
    np.save(scores_folder + "/cosine_scores_false_task3_merged", cosine_scores_false)
    np.save(scores_folder + "/sexist_similarity_task3_merged", sexist_similarity)
    np.save(scores_folder + "/non_sexist_similarity_task3_merged", non_sexist_similarity)
else:
    # Load cosine scores and similarities from saved npy files
    cosine_scores_true = np.load(scores_folder + "/cosine_scores_true_task3_merged.npy")
    cosine_scores_false = np.load(scores_folder + "/cosine_scores_false_task3_merged.npy")
    sexist_similarity = np.load(scores_folder + "/sexist_similarity_task3_merged.npy")
    non_sexist_similarity = np.load(scores_folder + "/non_sexist_similarity_task3_merged.npy")

# Binary classification
binary_preds = np.greater(sexist_similarity - k * non_sexist_similarity, epsilon)

print("Binary prediction: (k = " + str(k) + ", ε = " + str(epsilon) + ")" + "\n",
      classification_report(tweets_true_labels, binary_preds))

# Get the indices of the top 10 tweets with highest sexist similarity score:
top_sexist_idx = np.argsort(sexist_similarity)
# Get the indices of the top 10 tweets with LOWEST sexist similarity score:
low_sexist_idx = np.argsort(sexist_similarity)
# Get the indices of the top 10 tweets with highest non-sexist similarity score:
top_nonsexist_idx = np.argsort(non_sexist_similarity)
# Get the indices of the top 10 tweets with the lowest non-sexist similarity score:
low_nonsexist_idx = np.argsort(non_sexist_similarity)
# Get the indices of 20 tweets classified as sexist by the binary classifier
# (having a binary_preds value of True), which have a sexist similarity score
# close to the mean sexist similarity score for those tweets, within a threshold:
averagely_sexist_idx = np.where(np.abs(sexist_similarity -
                                       np.mean(sexist_similarity[np.where(binary_preds == 1)])) < 0.0008)[0]
# Likewise for non-sexist:
averagely_nonsexist_idx = np.where(np.abs(non_sexist_similarity -
                                          np.mean(non_sexist_similarity[np.where(binary_preds == 0)])) < 0.0008)[0]
# Check tweets 'near the threshold':
near_threshold_idx = np.where(np.abs(sexist_similarity - k * non_sexist_similarity - epsilon) < 0.001)[0]

# Indices of tweets misclassified as sexist in binary classifier but also having indices in top_sexist_idx:
misclassified_as_sexist_idx = np.intersect1d(np.where((binary_preds == 1) &
                                                      (binary_preds != tweets_true_labels))[0], top_sexist_idx[-70:])

misclassified_as_nonsexist_idx = np.intersect1d(np.where((binary_preds == 0) &
                                                         (binary_preds != tweets_true_labels))[0], top_nonsexist_idx[-200:])

print("\nTop 20 tweets with highest sexist similarity score:\n")
for idx in top_sexist_idx[-20:]:
    print(tweets[idx])
print("\nSome tweets with sexist similarity score close to the mean sexist "
      "similarity score of tweets classified sexist:\n")
for idx in averagely_sexist_idx[:20]:
    print(tweets[idx])
print("\nTop 20 tweets with lowest sexist similarity score:\n")
for idx in low_sexist_idx[:20]:
    print(tweets[idx])
print("\nTop 20 tweets with highest non-sexist similarity score:\n")
for idx in top_nonsexist_idx[-20:]:
    print(tweets[idx])
print("\nSome tweets with nonsexist similarity score close to the mean nonsexist"
      " similarity score of tweets classified nonsexist:\n")
for idx in averagely_nonsexist_idx[:20]:
    print(tweets[idx])
print("\nTop 20 tweets with lowest non-sexist similarity score:\n")
for idx in low_nonsexist_idx[:20]:
    print(tweets[idx])
print("\nSome tweets near the threshold:\n")
for idx in near_threshold_idx[:20]:
    print(tweets[idx])
print("\nSome tweets misclassified as sexist with high sexist similarity score:\n")
for idx in misclassified_as_sexist_idx[:20]:
    print(tweets[idx])
print("\nSome tweets misclassified as nonsexist with high nonsexist similarity score:\n")
for idx in misclassified_as_nonsexist_idx[:20]:
    print(tweets[idx])

# Compute ROC curve for binary classification:
true_probability = sexist_similarity - k * non_sexist_similarity
# Normalize to [0,1] range, to represent a probability:
true_probability = (true_probability - true_probability.min()) / (true_probability.max() - true_probability.min())

# Compute ROC:
fpr, tpr, thresholds = roc_curve(tweets_true_labels, true_probability, pos_label=True)
# fpr, tpr, thresholds = roc_curve(tweets_true_labels, true_probability, pos_label=False)
precision, recall, thresholds_ = precision_recall_curve(tweets_true_labels, true_probability, pos_label=True)
# precision, recall, thresholds_ = precision_recall_curve(tweets_true_labels, true_probability, pos_label=False)
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

# Plot ROC curve and show AUC:
plt.title('ROC Curve (sexist as positive class)')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--', label='Random guessing')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Plot Precision-Recall curve and show AUC:
plt.title('Precision-Recall Curve (sexist as positive class)')
plt.plot(recall, precision, 'b', label='AUC = %0.2f' % pr_auc)
plt.plot([0, 1], [1269/12753, 1269/12753], 'r--', label='Random guessing')
plt.legend(loc='upper right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
