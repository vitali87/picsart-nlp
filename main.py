import gensim
import nltk

sentence = "I really enjoy studying at Picsart Academy"

# download the required files
nltk.download('punkt')

tokens = nltk.word_tokenize(sentence)

# # download the required files
nltk.download('averaged_perceptron_tagger')

# part-of-speech tagging
tagged = nltk.pos_tag(tokens)

# to see all tags
nltk.download('tagsets')
nltk.help.upenn_tagset()

# Identify named entities
nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)

# Display the tree
entities.draw()

nltk.download('brown')
from nltk.corpus import brown

brown.raw()
brown.words()
brown.sents()
brown.paras(categories='reviews')

nltk.download('genesis')
nltk.corpus.genesis.words()

# Demonstrate word embedding using Gensim ()
from nltk.test.gensim_fixt import setup_module

setup_module()

train_set = brown.sents()[:10000]

model = gensim.models.Word2Vec(train_set)
len(model.wv['university'])

model.wv.similarity('university', 'education')
model.wv.similarity('university', 'school')

from nltk.data import find

nltk.download('word2vec_sample')
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
new_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# Searching word similarities
new_model.similarity("king", "queen")
new_model.most_similar(positive=['king'], topn=3)
