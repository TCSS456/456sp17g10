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

# Environmental Variables
os.environ['STANFORD_PARSER'] = './stanford-parser-full/'
os.environ['STANFORD_MODELS'] = './stanford-parser-full/'

MODEL_PATH = "./stanford-parser-full/englishPCFG.ser.gz";

class multi_document_dependency_ranker:
  def __init__(self):
    self.dep_parser = StanfordDependencyParser(model_path=MODEL_PATH);
    self.dep_parser.java_options = '-mx3052m'

  def __build_graph(self, nodes):
    graph = nx.Graph()
    
    graph.add_nodes_from(nodes);
    nodePairs = list(itertools.combinations(nodes, 2));

    for pair in nodePairs:
        first = pair[0];
        second = pair[1];
        weight = self.__calculateWeight(first, second);
        graph.add_edge(first, second, weight=weight);
    return graph;
  
  def __calculateWeight(self, first, second):
    #sentiment
    #frequency
    return 1;

  def rank(self, documents):
    #parse documents to nodes/dependencies
    (nodes,frequencies) = self.__parse(documents);
    graph = self.__build_graph(nodes);

    #debug view graph
    nx.draw(graph, with_labels = True);
    plt.show();

    ranked_dependencies = nx.pagerank(graph, weight='weight');
    sorted_dependencies = sorted(ranked_dependencies, key=ranked_dependencies.get, reverse=True);
    one_third = len(nodes);
    key_dependenceies = sorted_dependencies[0:one_third+1];

  def __is_hashtag(self, word):
    return '#' in word;

  def __parse(self, documents):
    stoplist = stopwords.words('english')
    results = self.dep_parser.raw_parse_sents(documents);
    word_set = set();
    morph_set = set();
    dependece_frequencies = {};
    morph_dependence_frequencies = {};
    for result in results:
      for graph in result:
        for i in graph.nodes:
          node = graph.nodes[i];
          if 'word' in node.keys() and node['word'] != None:
            word = node['word'];
            if not word in stoplist and not self.__is_hashtag(word):
              word_set.add(word);
        for triple in graph.triples():
          first = triple[0][0];
          second = triple[2][0];
          for word in (first, second):
            if word not in stoplist:
              if word in dependece_frequencies and not self.__is_hashtag(word):
                dependece_frequencies[word] += 1;
              else:
                dependece_frequencies[word] = 1;
    
    for word in dependece_frequencies:
      morph_word = wn.morphy(word);
      if morph_word != None:
        morph_set.add(morph_word);
        if morph_word in morph_dependence_frequencies:
          morph_dependence_frequencies[morph_word] += dependece_frequencies[word];
        else:
          morph_dependence_frequencies[morph_word] = dependece_frequencies[word];
      else:
        morph_dependence_frequencies[word] = dependece_frequencies[word];

    #calculate word relational frequencies
    pprint(dependece_frequencies);
    pprint(morph_dependence_frequencies);
    
    
    return (list(word_set), dependece_frequencies);
    

#test data

documents = ["I have a cat. The cat is cool. #coolcat", "He has a cat. His cat is cool.", "They have a cat. Their cat is cool."];

#convert all to lower case
for i, document in enumerate(documents):
  documents[i] = document.lower();

ranker = multi_document_dependency_ranker();
ranker.rank(documents);