import gensim
import nltk
from nltk.test.gensim_fixt import setup_module
from nltk.data import find
from nltk.corpus import brown

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np

import matplotlib.pyplot as plt

sentence = "I really enjoy studying at Picsart Academy"

# download the required files
nltk.download("punkt")

tokens = nltk.word_tokenize(sentence)

# # download the required files
nltk.download("averaged_perceptron_tagger")

# part-of-speech tagging
tagged = nltk.pos_tag(tokens)

# to see all tags
nltk.download("tagsets")
nltk.help.upenn_tagset()

# Identify named entities
nltk.download("maxent_ne_chunker")
nltk.download("words")
entities = nltk.chunk.ne_chunk(tagged)

# Display the tree
entities.draw()

nltk.download("brown")

brown.raw()
brown.words()
brown.sents()
brown.paras(categories="reviews")

nltk.download("genesis")
nltk.corpus.genesis.words()

# Demonstrate word embedding using Gensim ()
setup_module()

train_set = brown.sents()[:10000]

model = gensim.models.Word2Vec(train_set)
len(model.wv["university"])

model.wv.similarity("university", "education")
model.wv.similarity("university", "school")

nltk.download("word2vec_sample")
word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# Searching word similarities
w2v.similarity("king", "queen")
w2v.most_similar(positive=["king"], topn=10)

w2v.most_similar(positive=["academy"], topn=10)

# Words that do not exist in the training set
w2v.doesnt_match(sentence.split())

# Some interesting stuff
# What is the output of ‘King - Man + Woman’ ?
w2v.most_similar(positive=["woman", "king"], negative=["man"], topn=1)

# ‘Russia - Moscow + London’
w2v.most_similar(positive=["London", "Russia"], negative=["Moscow"], topn=1)
# ‘Russia - Moscow + Paris + London’
w2v.most_similar(positive=["Paris", "London", "Russia"], negative=["Moscow"], topn=3)

# Visualise Word Embeddings
labels = []
count = 0
max_count = 1000
X = np.zeros(shape=(max_count, len(w2v["academy"])))

for term in w2v.index_to_key:
    X[count] = w2v[term]
    labels.append(term)
    count += 1
    if count >= max_count:
        break

# Use PCA first to reduce to ~50 dimensions

pca = PCA(n_components=50)
X_50 = pca.fit_transform(X)

# Using TSNE to further reduce to 2 dimensions

model_tsne = TSNE(n_components=2, random_state=0)
Y = model_tsne.fit_transform(X_50)

# Show the scatter plot
plt.scatter(Y[:, 0], Y[:, 1], 20)

# Add labels
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points", size=10)
plt.show()
