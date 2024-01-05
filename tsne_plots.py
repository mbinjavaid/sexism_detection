import os
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Which plot to generate (can be 1, 2, 3, 4, 5, or 6) - as in the presentation.
which_plot = 1

if which_plot == 1:
    n = 878
elif which_plot == 2:
    n = 2000
elif which_plot == 3 or which_plot == 4:
    n = 338
elif which_plot == 5 or which_plot == 6:
    n = 1269
else:
    raise ValueError("Invalid value for which_plot (possible values are 1 till 6)")

gen_folder = "gpt"
dataset_folder = "Call me sexist but"
gen_file = "gpt_tweets.csv"
gen_scales_file = "ask3_sexist_tweetandscale.csv"
data_file = "sexism_data.csv"

# The model to use for encoding the text:
sbert_model = 'all-MiniLM-L6-v2'

gen_path = os.path.join(gen_folder, gen_file)
data_path = os.path.join(dataset_folder, data_file)

gpt_tweets = pd.read_csv (gen_path)
gpt_sexist_scales = pd.read_csv(os.path.join(gen_folder, gen_scales_file))
dataset = pd.read_csv(data_path)

gpt_true = gpt_tweets[gpt_tweets['sexist'] == True]['tweet'].to_numpy()[0:n]
gpt_false = gpt_tweets[gpt_tweets['sexist'] == False]['tweet'].to_numpy()[0:n]
gpt_all = gpt_tweets['tweet'].to_numpy()[0:n]
gpt_sexist_scales = gpt_sexist_scales['out_scale'].to_numpy()[0:n]

orig_scales_mask_false = (dataset["dataset"] == "scales") & (dataset["sexist"] == False)
orig_scales_false = dataset.loc[orig_scales_mask_false, ["text"]]["text"].to_numpy()[0:n]

orig_scales_mask_true = (dataset["dataset"] == "scales") & (dataset["sexist"] == True)
orig_scales_true = dataset.loc[orig_scales_mask_true, ["text"]]["text"].to_numpy()[0:n]

orig_scales_all = dataset.loc[dataset["dataset"] == "scales", ["text"]]["text"].to_numpy()[0:n]

orig_tweets_mask_false = (dataset["dataset"] != "scales") & (dataset["sexist"] == False)
orig_tweets_false = dataset.loc[orig_tweets_mask_false, ["text"]]["text"].to_numpy()[0:n]

orig_tweets_mask_true = (dataset["dataset"] != "scales") & (dataset["sexist"] == True)
orig_tweets_true = dataset.loc[orig_tweets_mask_true, ["text"]]["text"].to_numpy()[0:n]

orig_tweets_all = dataset.loc[dataset["dataset"] != "scales", ["text"]]["text"].to_numpy()[0:n]

for i in range(len(gpt_true)):
    # Remove newline characters:
    gpt_true[i] = gpt_true[i].replace('\n', ' ')
    # Remove trailing whitespace:
    gpt_true[i] = gpt_true[i].strip()

for i in range(len(gpt_false)):
    # Remove newline characters:
    gpt_false[i] = gpt_false[i].replace('\n', ' ')
    # Remove trailing whitespace:
    gpt_false[i] = gpt_false[i].strip()

for i in range(len(gpt_sexist_scales)):
    # Remove newline characters:
    gpt_sexist_scales[i] = gpt_sexist_scales[i].replace('\n', ' ')
    # Remove trailing whitespace:
    gpt_sexist_scales[i] = gpt_sexist_scales[i].strip()

for i in range(len(gpt_all)):
    # Remove newline characters:
    gpt_all[i] = gpt_all[i].replace('\n', ' ')
    # Remove trailing whitespace:
    gpt_all[i] = gpt_all[i].strip()

# Remove URLs, retweet tag at beginning, mention tag from tweets,
# since they do not really provide information about the text itself:
mention_tag = re.compile(r'\bMENTION\d+\b')
for i in range(len(orig_tweets_all)):
    orig_tweets_all[i] = re.sub(r'http\S*:{1}/{2}\S+', '', orig_tweets_all[i]).strip()
    orig_tweets_all[i] = orig_tweets_all[i].removeprefix('RT')
    orig_tweets_all[i] = re.sub(mention_tag, '', orig_tweets_all[i]).strip()
    # Remove newline characters:
    orig_tweets_all[i] = orig_tweets_all[i].replace('\n', ' ')
    # Remove trailing whitespace:
    orig_tweets_all[i] = orig_tweets_all[i].strip()

for i in range(len(orig_tweets_false)):
    orig_tweets_false[i] = re.sub(r'http\S*:{1}/{2}\S+', '', orig_tweets_false[i]).strip()
    orig_tweets_false[i] = orig_tweets_false[i].removeprefix('RT')
    orig_tweets_false[i] = re.sub(mention_tag, '', orig_tweets_false[i]).strip()
    # Remove newline characters:
    orig_tweets_false[i] = orig_tweets_false[i].replace('\n', ' ')
    # Remove trailing whitespace:
    orig_tweets_false[i] = orig_tweets_false[i].strip()

for i in range(len(orig_tweets_true)):
    orig_tweets_true[i] = re.sub(r'http\S*:{1}/{2}\S+', '', orig_tweets_true[i]).strip()
    orig_tweets_true[i] = orig_tweets_true[i].removeprefix('RT')
    orig_tweets_true[i] = re.sub(mention_tag, '', orig_tweets_true[i]).strip()
    # Remove newline characters:
    orig_tweets_true[i] = orig_tweets_true[i].replace('\n', ' ')
    # Remove trailing whitespace:
    orig_tweets_true[i] = orig_tweets_true[i].strip()

# Initialize the SBERT model:
model = SentenceTransformer(sbert_model)


# # Compute the embeddings:
print("Computing embeddings ...")
gpt_true_embeddings = model.encode(gpt_true, convert_to_tensor=True).cpu().detach().numpy()
gpt_false_embeddings = model.encode(gpt_false, convert_to_tensor=True).cpu().detach().numpy()

gpt_all_embeddings = model.encode(gpt_all, convert_to_tensor=True).cpu().detach().numpy()

gpt_sexist_embeddings = model.encode(gpt_sexist_scales, convert_to_tensor=True).cpu().detach().numpy()

orig_scales_true_embeddings = model.encode(orig_scales_true, convert_to_tensor=True).cpu().detach().numpy()
orig_scales_false_embeddings = model.encode(orig_scales_false, convert_to_tensor=True).cpu().detach().numpy()

orig_scales_all_embeddings = model.encode(orig_scales_all, convert_to_tensor=True).cpu().detach().numpy()

orig_tweets_true_embeddings = model.encode(orig_tweets_true, convert_to_tensor=True).cpu().detach().numpy()
orig_tweets_false_embeddings = model.encode(orig_tweets_false, convert_to_tensor=True).cpu().detach().numpy()

orig_tweets_all_embeddings = model.encode(orig_tweets_all, convert_to_tensor=True).cpu().detach().numpy()


# # Make TSNE plot for embeddings:
tsne = TSNE(n_components=2, random_state=123)
all_embeddings1 = np.concatenate((gpt_true_embeddings, gpt_false_embeddings, gpt_sexist_embeddings,
                                 orig_scales_true_embeddings, orig_scales_false_embeddings,
                                 orig_tweets_true_embeddings, orig_tweets_false_embeddings))

all_embeddings2 = np.concatenate((gpt_all_embeddings, gpt_sexist_embeddings, orig_scales_all_embeddings, orig_tweets_all_embeddings))

all_embeddings3 = np.concatenate((gpt_true_embeddings, gpt_false_embeddings, gpt_sexist_embeddings,
                                 orig_tweets_true_embeddings, orig_tweets_false_embeddings))

print("Computing TSNE ...")
if which_plot != 1 and which_plot != 5 and which_plot != 6:
    all_embeddings_tsne = tsne.fit_transform(all_embeddings1)
elif which_plot == 1:
    all_embeddings_tsne = tsne.fit_transform(all_embeddings2)
elif which_plot == 5 or which_plot == 6:
    all_embeddings_tsne = tsne.fit_transform(all_embeddings3)
else:
    raise ValueError("Invalid value for which_plot (possible values are 1 till 6)")

# Use matplotlib to plot the TSNE, using different colors for the embeddings.
fig, ax = plt.subplots()

if which_plot == 6:
    ax.scatter(all_embeddings_tsne[3*n:4*n, 0], all_embeddings_tsne[3*n:4*n, 1], alpha=0.15, s=40, c='r', label="Original Tweets (True)")
    ax.scatter(all_embeddings_tsne[0:n, 0], all_embeddings_tsne[0:n, 1], alpha=0.15, s=40, c='darkorange', label="GPT Tweet-like Scales (True)")
    ax.scatter(all_embeddings_tsne[4*n:5*n, 0], all_embeddings_tsne[4*n:5*n, 1], alpha=0.15, s=40, c='b', label="Original Tweets (False)")

if which_plot == 5:
    ax.scatter(all_embeddings_tsne[3 * n:4 * n, 0], all_embeddings_tsne[3 * n:4 * n, 1], alpha=0.15, s=40, c='r', label="Original Tweets (True)")
    ax.scatter(all_embeddings_tsne[4 * n:5 * n, 0], all_embeddings_tsne[4 * n:5 * n, 1], alpha=0.15, s=40, c='b', label="Original Tweets (False)")

if which_plot == 4:
    ax.scatter(all_embeddings_tsne[3*n:4*n, 0], all_embeddings_tsne[3*n:4*n, 1], alpha=0.3, s=70, c='r', label="Original Scales (True)")
    ax.scatter(all_embeddings_tsne[2*n:3*n, 0], all_embeddings_tsne[2*n:3*n, 1], alpha=0.3, s=70, c='darkorange', label="GPT Regular Scales (True)")
    ax.scatter(all_embeddings_tsne[4*n:5*n, 0], all_embeddings_tsne[4*n:5*n, 1], alpha=0.3, s=70, c='b', label="Original Scales (False)")

if which_plot == 3:
    ax.scatter(all_embeddings_tsne[3 * n:4 * n, 0], all_embeddings_tsne[3 * n:4 * n, 1], alpha=0.3, s=70, c='r', label="Original Scales (True)")
    ax.scatter(all_embeddings_tsne[4 * n:5 * n, 0], all_embeddings_tsne[4 * n:5 * n, 1], alpha=0.3, s=70, c='b', label="Original Scales (False)")

if which_plot == 2:
    ax.scatter(all_embeddings_tsne[0:n, 0], all_embeddings_tsne[0:n, 1], alpha=0.3, s=15, c='r', label="GPT Tweet-like Scales (True)")
    ax.scatter(all_embeddings_tsne[2*n:3*n, 0], all_embeddings_tsne[2*n:3*n, 1], alpha=0.3, s=15, c='darkorange', label="GPT Regular Scales (True)")
    ax.scatter(all_embeddings_tsne[n:2*n, 0], all_embeddings_tsne[n:2*n, 1], alpha=0.3, s=15, c='b', label="GPT Tweet-like Scales (False)")

if which_plot == 1:
    ax.scatter(all_embeddings_tsne[0:n, 0], all_embeddings_tsne[0:n, 1], alpha=0.3, s=40, c='r', label="GPT Tweet-like Scales")
    ax.scatter(all_embeddings_tsne[n:2*n, 0], all_embeddings_tsne[n:2*n, 1], alpha=0.3, s=40, c='darkorange', label="GPT Regular Scales")
    ax.scatter(all_embeddings_tsne[2*n:3*n, 0], all_embeddings_tsne[2*n:3*n, 1], alpha=0.3, s=40, c='b', label="Original Scales")
    ax.scatter(all_embeddings_tsne[3*n:4*n, 0], all_embeddings_tsne[3*n:4*n, 1], alpha=0.3, s=40, c='g', label="Original Tweets")

ax.legend(loc='upper right')
plt.show()
