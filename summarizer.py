import pandas as pd
import numpy as np
import spacy
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import nltk
from transformers import pipeline

class Summarizer:
  def __init__(self, album_name):
    self.album_name = album_name
    self.reviews_df = pd.read_csv("./data/" + album_name + "_reviews.csv")
    self.nlp = spacy.load('en_core_web_lg')

  def token_filter(self, token):
    if not token.is_stop: # Only keep tokens with vectors and that are not stopwords
      if token.pos_ in ['NOUN', 'ADJ', 'PROPN']: # Only keep nouns and adjectives (Maybe proper nouns too?)
        return True
    return False
  
  def preprocess(self):
    print("Preprocessing reviews for " + self.album_name)
    # Remove punctuation and lowercase
    print("Removing punctuation and lowercasing")
    self.reviews_df['processed_review'] = self.reviews_df['review'].str.replace('[^\w\s]','').str.lower()
    # Filter tokens
    print("Filtering tokens")
    self.reviews_df['processed_review'] = self.reviews_df['processed_review'].apply(lambda x: [token for token in self.nlp(x) if self.token_filter(token)])
    # Drop empty reviews
    print("Dropping empty reviews")
    self.reviews_df = self.reviews_df[self.reviews_df['processed_review'].map(len) > 0]
    # Generate document vectors
    print("Generating document vectors")
    self.reviews_df["vector"] = self.reviews_df["processed_review"].apply(lambda x: np.mean([token.vector for token in x], axis=0))
    # Remove reviews with zero vectors
    print("Removing reviews with zero vectors")
    self.reviews_df = self.reviews_df[self.reviews_df['vector'].map(np.any)]
    # Save processed reviews
    print("Saving processed reviews")
    self.reviews_df.to_csv("./data/" + self.album_name + "_processed_reviews.csv", index=False)
    return self.reviews_df
  
  def cluster(self):
    print("Clustering reviews for " + self.album_name)
    cluster = AgglomerativeClustering(**{'n_clusters': None, 'distance_threshold': 0.3, 'linkage': 'complete', 'metric': 'cosine', 'compute_full_tree': True})
    cluster.fit(self.reviews_df['vector'].tolist())
    self.reviews_df['cluster'] = cluster.labels_
    self.filtered_clusters = self.reviews_df.groupby("cluster").agg({"score": "var", "review": "count"}).sort_values("review", ascending=False)
    self.filtered_clusters = self.filtered_clusters[(self.filtered_clusters["review"] > self.filtered_clusters["review"].mean()) & (self.filtered_clusters["score"] <= self.filtered_clusters["score"].quantile(0.25))]
    return self.filtered_clusters
  
  def position_rank(self, cluster_df, key_phrase):
    doc = self.nlp(' '.join(cluster_df['review'].values.tolist()))

    word_count = defaultdict(lambda: defaultdict(lambda: 0))
    window_size = 4

    words = [token.lemma_ for token in doc if self.token_filter(token)]

    # Count co-occurences in a window
    for i, token in enumerate(words):
      for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
        if i != j:
          word_count[token.lower()][words[j].lower()] += 1

    # Create graph with nodes as words and edges as co-occurences
    G = nx.Graph()

    for word, count in word_count.items():
      for co_word, co_count in count.items():
        G.add_edge(word, co_word, weight=co_count)

    # Calculate positional rank for each word
    overall_word_position_rank = defaultdict(lambda: 0)

    for review in cluster_df['processed_review']:
      review_position_rank = defaultdict(lambda: 0)
      for i, token in enumerate(review):
        review_position_rank[token.text] += 1 / (i + 1)
      for word, rank in review_position_rank.items():
        overall_word_position_rank[word] = (overall_word_position_rank[word] + rank) / 2

    overall_word_position_rank

    factor = 1.0 / sum(overall_word_position_rank.values())

    normalized_overall_word_position_rank = {k: v * factor for k, v in overall_word_position_rank.items()}

    # Calculate word scores

    try:  
      word_scores = nx.pagerank(G, personalization=normalized_overall_word_position_rank, weight='weight')
    except:
      word_scores = nx.pagerank(G, weight='weight')

    # Generate key phrases
    final_scores = {}

    if key_phrase:
      score = 0
      phrase = []
      phrase_tags = []

      for token in doc:
        if len(phrase) == 3:
          if 'NOUN' in phrase_tags or 'PROPN' in phrase_tags:
            final_scores[' '.join(phrase).lower()] = score
          score = 0
          phrase = []
          phrase_tags = []

        if token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
          score += word_scores.get(token.lemma_.lower(), 0)
          phrase.append(token.lemma_)
          phrase_tags.append(token.pos_)
        else:
          if len(phrase) > 0 and ('NOUN' in phrase_tags or 'PROPN' in phrase_tags):
            final_scores[(' '.join(phrase)).lower()] = score
          score = 0
          phrase = []

      if 'NOUN' in phrase_tags or 'PROPN' in phrase_tags:
        final_scores[' '.join(phrase).lower()] = score
        score = 0
        phrase = []
        phrase_tags = []

      return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    else:
      sentence_scores = []

      sentences = []

      for review in cluster_df["review"]:
          sentences.extend(nltk.sent_tokenize(review))

      for sentence in sentences:
          score = 0
          word_count = 1
          for token in self.nlp(sentence):
              word_score = word_scores.get(token.text)
              if word_score:
                  score += word_score
                  word_count += 1
            
          sentence_scores.append(score/word_count)

      sentence_scores_df = pd.DataFrame({'sentence': sentences, 'score': sentence_scores})

      return sentence_scores_df.sort_values(by='score', ascending=False).head(3)['sentence'].values.tolist()
    
  def generate_key_phrases(self, key_phrase=False):
    key_phrases = []
    
    print("Generating key phrases for " + self.album_name)

    for cluster in self.filtered_clusters.index:
      cluster_df = self.reviews_df[self.reviews_df['cluster'] == cluster]
      phrases = self.position_rank(cluster_df, key_phrase)
      if phrases:
        key_phrases.extend(phrases)

    return key_phrases
  
  def generate_summary(self):
    sentences = self.generate_key_phrases()
    sentences = list(set(sentences))
    text = '. '.join(sentences)
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
    return summary

if __name__ == "__main__":
  album_name = "bringing-it-all-back-home"
  summarizer = Summarizer(album_name)
  summarizer.preprocess()
  summarizer.cluster()
  print(summarizer.generate_summary())