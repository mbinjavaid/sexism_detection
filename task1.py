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
k = 0.6
epsilon = 0.13

# Directories for the datasets (should contain sexism_data.csv and sexism_annotations.csv):
dataset_folder = "Call me sexist but"

# # File names:
# File containing 'Call me sexist but' dataset (binary labels):
data_file = "sexism_data.csv"
# File containing 'Call me sexist but' dataset (fine-grained labels)
annot_file = "sexism_annotations.csv"

# The model to use for encoding the text: (can also try out different ones)
sbert_model = 'all-MiniLM-L6-v2'

data_path = os.path.join(dataset_folder, data_file)
annot_path = os.path.join(dataset_folder, annot_file)

data = pd.read_csv (data_path)
annot = pd.read_csv(annot_path)


# # Data pre-processing:

# Define selection criteria to extract scale items and tweets:
scales_mask_true = (data["dataset"] == "scales") & (data["sexist"] == True)
scales_mask_false = (data["dataset"] == "scales") & (data["sexist"] == False)

tweets_mask = (data["dataset"] != "scales")

# Get scale items and IDs:
scales_true = data.loc[scales_mask_true, ["id", "text"]]
scales_false = data.loc[scales_mask_false, ["id", "text"]]
scales_true_ids = scales_true["id"].to_numpy()
scales_false_ids = scales_false["id"].to_numpy()

scales_true = scales_true["text"].to_numpy()

scales_false = scales_false["text"].to_numpy()

# Get (non-scale item) tweets and IDs:
tweets_ = data.loc[tweets_mask, ["id", "text", "sexist"]]
tweets_ = tweets_.drop_duplicates(subset='id', keep="first")
tweets_ids = tweets_["id"].to_numpy()
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

# Get categories of sexist content (1-4 are types of sexism, 5=can't tell, 6=not sexist) and phrasing (1-3):
# There are multiple entries for each id, so we reduce each id's category to the most frequent one (the mode)
categories = annot[["id", "phrasing", "content"]]
# Picks the mode (most frequent category) for phrasing and content categories for tweets:
mode_categories = categories[['phrasing', 'content']].groupby(categories['id']).apply(
    lambda g: g.mode()).reset_index()[['id', 'phrasing', 'content']]
# Remove duplicate ID rows:
mode_categories = mode_categories.drop_duplicates(subset='id', keep="first")
# mode_categories = mode_categories.dropna()

# Filter out only those ID rows which are present in sexism_data.csv, for both tweets and scale items
tweets_categories = mode_categories[mode_categories['id'].isin(tweets_['id'])]
# Get only the tweets that have been annotated - will need this for fine-grained testing later:
tweets_intersection = tweets_[tweets_['id'].isin(tweets_categories['id'])]
scales_true_categories = mode_categories[mode_categories['id'].isin(scales_true_ids)]
scales_false_categories = mode_categories[mode_categories['id'].isin(scales_false_ids)]

# filter rows based on ID presence in scales_true_ids and scales_false_ids and maintain order
true_mode_categories = scales_true_categories[scales_true_categories['id'].isin(scales_true_ids)].iloc[np.argsort(np.argsort(scales_true_ids))]
false_mode_categories = scales_false_categories[scales_false_categories['id'].isin(scales_false_ids)].iloc[np.argsort(np.argsort(scales_false_ids))]

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

    np.save(scores_folder + "/cosine_scores_true_only_original_scales", cosine_scores_true)
    np.save(scores_folder + "/cosine_scores_false_only_original_scales", cosine_scores_false)
    np.save(scores_folder + "/sexist_similarity_only_original_scales", sexist_similarity)
    np.save(scores_folder + "/non_sexist_similarity_only_original_scales", non_sexist_similarity)
else:
    # Load cosine scores and similarities from saved npy files
    cosine_scores_true = np.load(scores_folder + "/cosine_scores_true_only_original_scales.npy")
    cosine_scores_false = np.load(scores_folder + "/cosine_scores_false_only_original_scales.npy")
    sexist_similarity = np.load(scores_folder + "/sexist_similarity_only_original_scales.npy")
    non_sexist_similarity = np.load(scores_folder + "/non_sexist_similarity_only_original_scales.npy")

# # Binary classification of tweets:

binary_preds = np.greater(sexist_similarity - k * non_sexist_similarity, epsilon)
print("Binary prediction: (k = " + str(k) + ", ε = " + str(epsilon) + ")" + "\n",
      classification_report(tweets_true_labels, binary_preds))

# # Compute metrics for baseline binary predictions (always predict tweets to be non-sexist, the majority class):
# baseline_preds = np.full(len(binary_preds), False)
# print("Binary prediction (baseline):\n", classification_report(tweets_true_labels, baseline_preds))

# Compute ROC curve for binary classification:
true_probability = sexist_similarity - k * non_sexist_similarity
# Normalize to [0,1] range, to represent a probability:
true_probability = (true_probability - true_probability.min()) / (true_probability.max() - true_probability.min())

# # Notable tweets:

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


# # Fine-grained classification of tweets:

# Compute average similarity of each tweet to the phrasing/content label categories of scale items
# get an array of 3 (6) values for each tweet, then see which is maximum, return that category prediction.

tweets_phrasing_preds = np.zeros(len(tweets_intersection))
tweets_content_preds = np.zeros(len(tweets_intersection))

# Cache the results of loc operations to speed up repeated look-ups later:
true_mode_categories_map = {id_val: (phrasing_val, content_val)
                            for id_val, phrasing_val, content_val in
                            zip(true_mode_categories['id'], true_mode_categories['phrasing'],
                                true_mode_categories['content'])}
false_mode_categories_map = {id_val: (phrasing_val, content_val)
                             for id_val, phrasing_val, content_val in
                             zip(false_mode_categories['id'], false_mode_categories['phrasing'],
                                 false_mode_categories['content'])}

for i in range(len(tweets_intersection)):
    phrasing = np.zeros(3)
    content = np.zeros(6)
    # Compute phrasing and content for true scales
    for j in range(len(scales_true)):
        id_val = scales_true_ids[j]
        if id_val in true_mode_categories_map:
            phrasing_idx, content_idx = true_mode_categories_map[id_val]
            score = cosine_scores_true[i][j]
            phrasing[int(phrasing_idx-1)] += score
            content[int(content_idx-1)] += score

    # Compute phrasing and content for false scales
    for k in range(len(scales_false)):
        id_val = scales_false_ids[k]
        if id_val in false_mode_categories_map:
            phrasing_idx, content_idx = false_mode_categories_map[id_val]
            score = cosine_scores_false[i][k]
            phrasing[int(phrasing_idx-1)] += score
            content[int(content_idx-1)] += score

    tweets_phrasing_preds[i] = np.argmax(phrasing) + 1
    tweets_content_preds[i] = np.argmax(content) + 1

# Fine-grained prediction metrics:
print("Fine-grained prediction (phrasing category):\n",
      classification_report(tweets_categories['phrasing'].to_numpy().astype(int), tweets_phrasing_preds.astype(int)))
# Compute metrics for baseline phrasing category predictions (always predict tweets in phrasing category 3):
baseline_phrasing_preds = np.full(len(tweets_phrasing_preds), 3).astype(int)
print("Fine-grained baseline prediction (phrasing category)):\n",
      classification_report(tweets_categories['phrasing'].to_numpy().astype(int), baseline_phrasing_preds))

print("Fine-grained prediction (content category):\n",
      classification_report(tweets_categories['content'].to_numpy().astype(int), tweets_content_preds.astype(int)))
# Compute metrics for baseline content category predictions (always predict tweets in content category 6):
baseline_content_preds = np.full(len(tweets_content_preds), 6).astype(int)
print("Fine-grained baseline prediction (content category)):\n",
      classification_report(tweets_categories['content'].to_numpy().astype(int), baseline_content_preds))


# # Display ROC and Precision-Recall curves for binary classification:
# Compute ROC:
fpr, tpr, thresholds = roc_curve(tweets_true_labels, true_probability, pos_label=True)
precision, recall, thresholds_ = precision_recall_curve(tweets_true_labels, true_probability, pos_label=True)
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
