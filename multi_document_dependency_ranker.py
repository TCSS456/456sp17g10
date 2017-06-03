import nltk;
from nltk.internals import find_jars_within_path;
from nltk.corpus import wordnet as wn;
from nltk.corpus import stopwords;

import matplotlib.pyplot as plt;
import itertools
import networkx as nx
import networkx.algorithms as nxaa


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

from datetime import datetime

from DependencyTool import DependencyTool
import SentenceGenerator as sg
import markovify

import pickle
import json

# Environmental Variables
os.environ['STANFORD_PARSER'] = './stanford-parser-full/'
os.environ['STANFORD_MODELS'] = './stanford-parser-full/'



MODEL_PATH = "./stanford-parser-full/englishPCFG.ser.gz";

class multi_document_dependency_ranker(object):
  def __init__(self):
    self.dep_parser = StanfordDependencyParser(model_path=MODEL_PATH);
    self.dep_parser.java_options = '-mx3052m'

    self.dependency_tool = DependencyTool();

    self.nodes = list();

  def __build_graph(self):
    graph = nx.Graph()
    for d in self.dependency_tool.dependencies('filtered'):
      name = '[["'+d[0][0]+'","'+d[0][1]+'"],"'+d[1]+'",["'+d[2][0]+'","'+d[2][1]+'"]]';
      graph.add_node(name, triple=d);

    complete_edges = itertools.combinations(graph.nodes(data=True), 2)

    for edge in complete_edges:
        first = edge[0][1]['triple'];
        second = edge[1][1]['triple'];
        weight = self.__calculateWeight(first, second);
        graph.add_edge(edge[0][0], edge[1][0], weight=weight);
    
    # remove isolated nodes
    graph.remove_nodes_from(nx.isolates(graph))
    remove = [node for node,degree in graph.degree().items() if degree == 0]
    graph.remove_nodes_from(remove)

    components = nx.connected_components(graph);

    i = 1;
    for component in components:
      print('[' + str(i) + ']');
      print('# elements: ' + str(len(component)));
      i = i + 1;

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

    weight = self.dependency_tool.frequency(first) * self.dependency_tool.frequency(second) * similarity;
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

    pprint(self.dependency_tool.dependencies('unfiltered'), open("nodes.txt", "w"));
    pprint(self.dependency_tool.dependency_frequencies('unfiltered'), open("node_frequencies.txt", "w"));
    pprint(self.dependency_tool.edge_frequencies('unfiltered'), open("edge_frequencies.txt", "w"));

    print("Building graph...")
    
    a = datetime.now();
    self.graph = self.__build_graph();
    b = datetime.now();
    c = b - a;
    print('[Build Graph Time] ' + str(c));

    #debug view graph
    #nx.draw(graph, with_labels = True);
    #plt.show();

    print("Ranking nodes...")
    self.ranked_nodes = nx.pagerank(self.graph, weight='weight');
    pprint(self.ranked_nodes, open("ranked-nodes.txt", "w"));
    sorted_nodes = sorted(self.ranked_nodes, key=self.ranked_nodes.get);

    pprint(self.ranked_nodes);

    print('Sorted Ranked Nodes:');
    pprint(sorted_nodes, open("sorted-ranked-nodes.txt", "w"));
    pprint("Complete...");

  def analyze(self):
    g = nx.DiGraph()
    added = set();
    score_tol = 0.0025;
    for n in self.ranked_nodes:
      score = self.ranked_nodes[n];
      if score > score_tol:
        triple = json.loads(n);
        a = wn.morphy(triple[0][0]) or triple[0][0];
        b = wn.morphy(triple[2][0]) or triple[2][0];
        if a not in added:
          g.add_node(a, pos=triple[0][1], weight=score);
          added.add(a);
        if b not in added:
          g.add_node(b, pos=triple[2][1], weight=score);
          added.add(b);
    for n in self.ranked_nodes:
      score = self.ranked_nodes[n];
      if score > score_tol:
        triple = json.loads(n);
        a = wn.morphy(triple[0][0]) or triple[0][0];
        b = wn.morphy(triple[2][0]) or triple[2][0];
        relationship = triple[1];
        g.add_edge(a, b, dep=relationship, weight=score);
    nx.draw(g, with_labels = True);
    plt.show();

    for e in g.edges(data=True):
      dep = e[2]['dep']
      str = "";
      if dep == 'amod':
        str = e[1] + " " + e[0];
      if dep == 'nmod':
        str = e[1] + " " + e[0] + "/NP";
      if dep == 'aux':
        str = e[1] + " " + e[0];
      if dep == 'dep':
        str = 'unknown dependency : %s, %s' % (e[0], e[1]);
      if not str == "":
        print(str);
      else:
        print(e);



    #print(list(g.in_degree_iter()));
    #print(list(g.out_degree_iter()));

    #ranked = nx.pagerank(g);
    #sorted_ = sorted(ranked, key=ranked.get, reverse=True);
    #l = list();
    #gw = "";
    #for i in sorted_:
    #  l.append((i, ranked[i]));




  def __is_hashtag(self, word):
    return '#' in word;

  def __clean_word(self, word):
    return wn.morphy(word) or word

  def __triples2graph(self, triples):
    G = nx.MultiDiGraph()
    added_words = set();
    for triple in triples:
      words = [triple[x][0] for x in (0,2)];
      base_words = [self.__clean_word(w) for w in words];
      poss = [triple[x][1] for x in (0,2)];
      rel = triple[1];
      for i in xrange(0, 1):
        (word, base, pos) = (words[i], base_words[i], poss[i]);
        if base not in added_words:
          added_words.add(base);
          G.add_node(base, word=list((word, pos)));
        else:
          for n in G.nodes(data=True):
            if n[0] == base:
              n[1]['word'].append((word, pos));
        G.add_edge(base_words[0], base_words[1], rel=rel);
    return G;

  def __parse_documents(self, documents):
    for document in documents:
      sents = nltk.sent_tokenize(document);
      if not os.path.isfile('parsed.pickle') or not os.path.isfile('graph.pickle'):
        print('Parsing...');
        a = datetime.now();
        results = self.dep_parser.raw_parse_sents(sents);
        b = datetime.now();
        c = b - a;
        print('[Parse Time] ' + str(c));
        print('Pickling parse...');
        all_triples = list();
        for result in results:
          for graph in result:
            triples = graph.triples();
            all_triples.extend(triples);
        self.dependency_tool.put_all(all_triples);
        self.G = self.__triples2graph(all_triples);
        
        pickle.dump(self.G, open('graph.pickle', 'wb'));
        pickle.dump(self.dependency_tool, open('parsed.pickle', 'wb'));
      else:
        print('Unpickling parse...');
        self.dependency_tool = pickle.load(open('parsed.pickle', 'rb'));
        self.G = pickle.load(open('graph.pickle', 'rb'));
    

#test data
alien_document = open('corpus.txt').read().replace('\r\n', ' ').lower();
documents = [alien_document];

#convert all to lower case
for i, document in enumerate(documents):
  documents[i] = document.lower();

if not os.path.isfile('resume.pickle'):
  ranker = multi_document_dependency_ranker();
  ranker.rank(documents);
  print('Dumping pickle...');
  pickle.dump(ranker, open('resume.pickle', 'wb'));
else:
  print('Loading pickle...');
  ranker = pickle.load(open('resume.pickle', 'rb'));
  ranker.analyze();
  print('Loaded.');

# markovify test
#pos_gen = sg.SentenceGenerator(alien_document);
#gen = markovify.combine([pos_gen, chain])
#print(gen.make_sentence());