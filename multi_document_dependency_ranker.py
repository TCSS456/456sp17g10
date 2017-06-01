import nltk;
from nltk.internals import find_jars_within_path;
from nltk.corpus import wordnet as wn;
from nltk.corpus import stopwords;

import matplotlib.pyplot as plt;
import itertools
import networkx as nx


# Stanford Parser Dependencies
import os;
from nltk.parse import stanford
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser

# Formatting Dependencies
from pprint import pprint;
from nltk.tree import *;

# Sentiment Analysis
from textblob import TextBlob

from DependencyTool import DependencyTool

# Environmental Variables
os.environ['STANFORD_PARSER'] = './stanford-parser-full/'
os.environ['STANFORD_MODELS'] = './stanford-parser-full/'



MODEL_PATH = "./stanford-parser-full/englishPCFG.ser.gz";

class multi_document_dependency_ranker:
  def __init__(self):
    self.dep_parser = StanfordDependencyParser(model_path=MODEL_PATH);
    self.dep_parser.java_options = '-mx3052m'

    self.dependency_tool = DependencyTool();

    self.nodes = list();
    self.dependency_frequencies = {};
    self.edge_frequencies = {};

  def __build_graph(self):
    graph = nx.Graph() # or digraph?
    join = lambda x:'(('+x[0][0]+','+x[0][1]+'),'+x[1]+',('+x[2][0]+','+x[2][1]+'))'
    graph.add_nodes_from(map(join, self.dependencies));

    complete_edges = itertools.combinations(self.dependencies, 2)

    for edge in complete_edges:
        first = edge[0];
        second = edge[1];
        weight = self.__calculateWeight(first, second);
        graph.add_edge(join(first), join(second), weight=weight);
    return graph;
  
  def __calculateWeight(self, first, second):
    a = wn.synsets(wn.morphy(first[0][0]) or first[0][0]);
    b = wn.synsets(wn.morphy(first[2][0]) or first[2][0]);
    y = wn.synsets(wn.morphy(second[0][0]) or second[0][0]);
    z = wn.synsets(wn.morphy(second[2][0]) or second[2][0]);

    similarity = 0;
    if a is None or b is None or y is None or z is None or len(a) == 0 or len(b) == 0 or len(y) == 0 or len(z) == 0:
      #nothing
      similarity = 0;
    else:
      similarity = similarity + (a[0].wup_similarity(y[0]) or 0);
      similarity = similarity + (a[0].wup_similarity(z[0]) or 0);
      similarity = similarity + (b[0].wup_similarity(y[0]) or 0);
      similarity = similarity + (b[0].wup_similarity(z[0]) or 0);
      similarity = similarity / float(4);

    #sentiment = abs(TextBlob(a).sentiment.polarity * TextBlob(b).sentiment.polarity * TextBlob(c)).sentiment.polarity * TextBlob(d)).sentiment.polarity);

    #cosine similarity todo
    #need overall sentiment

    weight = self.dependency_tool.frequency(first) * self.dependency_tool.frequency(first) * similarity;
    return weight;

  def __order(first, second):
    if first < second:
      a = first;
      b = second;
    else:
      a = second;
      b = first;
    return (a, b);

  def rank(self, documents):
    #parse documents
    self.__parse_documents(documents);

    pprint(self.dependencies, open("nodes.txt", "w"));
    pprint(self.dependency_frequencies, open("node_frequencies.txt", "w"));
    pprint(self.edge_frequencies, open("edge_frequencies.txt", "w"));

    print("Building graph...")
    graph = self.__build_graph();
    #debug view graph
    #nx.draw(graph, with_labels = True);
    #plt.show();

    print("Ranking nodes...")
    ranked_nodes = nx.pagerank(graph, weight='weight');
    sorted_nodes = sorted(ranked_nodes, key=ranked_nodes.get);

    print('Sorted Ranked Nodes:');
    pprint(sorted_nodes);

  def __is_hashtag(self, word):
    return '#' in word;

  def __parse_documents(self, documents):
    for document in documents:
      sents = nltk.sent_tokenize(document);
      results = self.dep_parser.raw_parse_sents(sents);
      for result in results:
        for graph in result:
          self.dependency_tool.put_all(graph.triples());
    
    self.dependencies = list(self.dependency_tool.dependencies());
    self.dependency_frequencies = self.dependency_tool.dependency_frequencies();
    self.edge_frequencies = self.dependency_tool.edge_frequencies();
    

#test data

documents = [open('corpus.txt').read().replace('\r\n', ' ')];

#convert all to lower case
for i, document in enumerate(documents):
  documents[i] = document.lower();

ranker = multi_document_dependency_ranker();
ranker.rank(documents);