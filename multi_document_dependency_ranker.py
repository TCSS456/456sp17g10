import nltk
from nltk.internals import find_jars_within_path
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import itertools
import networkx as nx
import networkx.algorithms as nxaa


# Stanford Parser Dependencies
import os
from nltk.parse import stanford
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser

# Formatting Dependencies
from pprint import pprint
from nltk.tree import *

# Sentiment Analysis
from textblob import TextBlob

from datetime import datetime
from math import floor
from math import pow

from DependencyTool import DependencyTool
import SentenceGenerator as sg
import markovify
from bs4 import BeautifulSoup

import pickle
import json
import copy
from operator import itemgetter

# twitter
import preprocessor as pre

# Environmental Variables
os.environ['STANFORD_PARSER'] = '../stanford-parser-full/'
os.environ['STANFORD_MODELS'] = '../stanford-parser-full/'

MODEL_PATH = "../stanford-parser-full/englishPCFG.ser.gz"

STOP_TYPES = ['det', 'aux', 'cop', 'neg', 'case', 'mark', 'nsubj', 'amod', 'nmod:poss', 'auxpass']
LEFT_NON_TERMINAL = ['nsubjpass']
ZAG_TYPE = ['advmod']
MAX_SENTENCE_LENGTH = 30

class multi_document_dependency_ranker(object):
  def __init__(self):
    self.dep_parser = StanfordDependencyParser(model_path=MODEL_PATH)
    self.dep_parser.java_options = '-mx3052m'

    self.dependency_tool = DependencyTool()

    self.nodes = list()

  def __triple2json(self, triple):
    return '[["'+triple[0][0]+'","'+triple[0][1]+'"],"'+triple[1]+'",["'+triple[2][0]+'","'+triple[2][1]+'"]]'

  def __build_graph(self):
    '''
    builds dependency graph from filtered (stopwords removed, morphy) dependencies
    O(n^2) complexity complete graph dependency comparison 
    '''
    graph = nx.Graph()
    for d in self.dependency_tool.dependencies('filtered'):
      name = self.__triple2json(d)
      morph1 = wn.synsets(wn.morphy(d[0][0]) or d[0][0])
      morph2 = wn.synsets(wn.morphy(d[2][0]) or d[2][0])
      graph.add_node(name, triple=d, morph1=morph1, morph2=morph2)
    complete_edges = itertools.combinations(graph.nodes(data=True), 2)

    start = datetime.now()
    n = 0
    nn = len(graph.nodes())
    total = nn*(nn-1)/2
    for edge in complete_edges:
      
      a = edge[0][1]['morph1']
      b = edge[0][1]['morph2']
      y = edge[1][1]['morph1']
      z = edge[1][1]['morph2']
      weight = self.__calculateWeight(a, b, y, z)

      if weight > 0:
        graph.add_edge(edge[0][0], edge[1][0], weight=weight)
      #debugging timer
      n = n + 1
      if n % 5000 == 0:
        end = datetime.now()
        frac = n / float(total)
        print('%d / %d iterations = %f, %s total time' % (n, total, frac, str(end-start)))
    
    # remove isolated nodes
    isolated = nx.isolates(graph)
    if len(isolated) > 0:
      print("removing %d isolated edges" % len(isolated))
      graph.remove_nodes_from(isolated)
      remove = [node for node,degree in graph.degree().items() if degree == 0]
      graph.remove_nodes_from(remove)
    
    # remove morph data
    for n in graph.nodes(data=True):
      n[1].pop('morph1', None)
      n[1].pop('morph2', None)

    #output number of graph components (clusters)
    components = nx.connected_components(graph)
    i = 1
    for component in components:
      print('component %d: #elements=%d' % (i, len(component)))
      i = i + 1

    return graph

  def __calculateWeight(self, a, b, y, z):
    '''
    Calculates edges weights between dependencies. Average similarity between each word in dependency.
    '''
    similarity = 0
    if not (a is None or b is None or y is None or z is None or len(a) == 0 or len(b) == 0 or len(y) == 0 or len(z) == 0):
      similarity = similarity + (a[0].wup_similarity(y[0]) or 0)
      similarity = similarity + (a[0].wup_similarity(z[0]) or 0)
      similarity = similarity + (b[0].wup_similarity(y[0]) or 0)
      similarity = similarity + (b[0].wup_similarity(z[0]) or 0)
      similarity = similarity / float(4)

    return similarity

  def __order(first, second):
    '''
    returns two strings as ordered tuple
    '''
    if first < second:
      a = first
      b = second
    else:
      a = second
      b = first
    return (a, b)

  def rank(self, documents):
    '''
    uses the pagerank algorithm to rank the document dependencies
    '''
    self.__parse_documents(documents)
    print("Building graph...")
    a = datetime.now()
    self.graph = self.__build_graph()
    b = datetime.now()
    c = b - a
    print('[Build Graph Time] %s' % c)
    print("Ranking nodes...")
    self.ranked_nodes = nx.pagerank(self.graph, weight='weight')
    print('Complete.')

  def analyze(self):
    '''
    generates a sumary paragraph for the given documents
    '''
    G = self.G
    roots = self.roots
    print("#roots: %d" % len(roots))
    all_sents = list()
    for n in roots:
      print(n)
      in_edges = G.in_edges(n)
      for in_edge in in_edges:
        sents = list()
        visited_nodes = {}
        left_sent = list()
        rel_chain = list()
        word_chain = list()
        prev_node = None
        cur_node = n
        cur_rel = 'root'
        score = [0]
        self.__find_sentences(G, roots, rel_chain, word_chain, visited_nodes, left_sent, prev_node, cur_node, cur_rel, sents, score)
        all_sents.extend(sents)
    MAX_ALLOWED_LENGTH = 30
    max_sent = None
    max_score = 0

    all_sents.sort()
    all_sents = list(all_sents for all_sents,_ in itertools.groupby(all_sents))
    sort = sorted(all_sents, key=itemgetter(-1))

    paragraph = list()
    for s in reversed(sort):
      if len(s) < 30:
        same = False
        for sentence in paragraph:
          if len(set(sentence).intersection(set(s))) > 5:
            same = True
            break
        if not same:
          paragraph.append(s)
    output = 'Summary:\n'
    for s in paragraph:
      output = output + ' '.join(s[:-1]).capitalize() + '. '
    print(output)

  
  def score_words(self):
    '''
    generates self.word_scores and self.min_word_scores
    for the markov chain calculation
    '''
    g = nx.DiGraph()
    added = set()
    for n in self.ranked_nodes:
      
      score = self.ranked_nodes[n]
      triple = json.loads(n)
      #sdistance = abs(TextBlob(triple[0][0] + ' ' + triple[2][0]).sentiment.polarity - self.overall_sentiment);
      score = score #/ sdistance;
      a = wn.morphy(triple[0][0]) or triple[0][0]
      b = wn.morphy(triple[2][0]) or triple[2][0]
      if a not in added:
        g.add_node(a, pos=triple[0][1], weight=score)
        added.add(a)
      if b not in added:
        g.add_node(b, pos=triple[2][1], weight=score)
        added.add(b)
    for n in self.ranked_nodes:
      score = self.ranked_nodes[n]
      triple = json.loads(n)
      a = wn.morphy(triple[0][0]) or triple[0][0]
      b = wn.morphy(triple[2][0]) or triple[2][0]
      relationship = triple[1]
      g.add_edge(a, b, dep=relationship, weight=score)

    nx.draw(g, with_labels = True)
    plt.show()

    o = g.out_degree()
    i = g.in_degree()
    t = {}
    for n in o:
      t[n] = o[n] + i[n]
    s = sorted(t, key=t.get)
    pprint(s)

    ranked = nx.pagerank(g)
    self.word_scores = ranked
    self.min_word_score = min(self.word_scores.values())

    

  def __find_sentences(self, G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, prev_node, cur_node, cur_rel, sents, score):
    '''
    G: the complete parse graph
    left_sent: current sentence up to current node
    cur_node: current node in graph
    sents: list of complete sentences
    '''
    #prevent infinite recursion
    if len(rel_chain) > MAX_SENTENCE_LENGTH:
      return False

    #prevent double words
    if prev_node == cur_node:
      return False

    #prevent more than two occurences of a word
    if self.__exceeds_occ_limit(cur_node, visited_nodes, 1):
      return False
    else:
      if cur_node in visited_nodes:
        c = visited_nodes[cur_node] + 1
      else:
        c = 1
      visited_nodes[cur_node] = c

    #prevent bigram repetition
    for i in xrange(0, len(left_sent) - 2):
      a = left_sent[i]
      b = left_sent[i + 1]
      if len(left_sent) >= 4:
        ca = left_sent.count(a)
        cb = left_sent.count(b)
        if ca > 1 and cb > 1:
          occ1 = [i for i,val in enumerate(left_sent) if val==a]
          occ2 = [i for i,val in enumerate(left_sent) if val==b]
          if occ1[-1] - occ1[-2] == occ2[-1] - occ2[-2]:
            # repeating 2+gram detected
            return False
          if len(occ2) > 2 and occ1[-1] - occ1[-2] == occ2[-2] - occ2[-3]:
            return False
          if len(occ1) > 2 and occ1[-2] - occ1[-3] == occ2[-1] - occ2[-2]:
            return False

    #stop divergent (score) paths
    if len(left_sent) >= 10 and score[0] < 0:
      return False

    word_chain = copy.deepcopy(word_chain)
    rel_chain = copy.deepcopy(rel_chain)
    cur_node_attrs = G.node[cur_node]
    rel_chain.append(cur_rel)
    word_chain.append(cur_node)

    if len(word_chain) > 2:
      triprob = self.dependency_tool.trigram_probability((word_chain[-3], rel_chain[-3]),(word_chain[-2], rel_chain[-2]),(word_chain[-1], rel_chain[-1]))
    else:
      triprob = 0

    smoothing = 0.1

    key = (prev_node, cur_rel, cur_node)
    freq = self.dependency_tool.frequency(key, 'lossy-unfiltered')
    
    if key in self.scoring:
      dep_score = self.scoring[key]
    else:
      dep_score = self.min_score

    word_score = self.min_word_score
    if cur_node in self.word_scores:
      word_score = word_score + self.word_scores[cur_node]

    score[0] = score[0] + dep_score * freq * word_score * (triprob + smoothing)
    
    out_degree = G.out_degree(cur_node)

    #choose mark only if good fit
    if cur_rel == 'mark' and not triprob > 0:
      return False

    if cur_rel in STOP_TYPES:
      left_sent.append(cur_node)
      return False
    else:
      out_edges = G.out_edges(cur_node, data=True)
      left_edges = list()
      right_edges = list()
      compound_edges = list()
      left_non_terminal_edges = list()
      cc = None
      conj_edge = None
      det = None
      case = None
      ccomp = None
      mark = None
      aux = None
      neg = None
      cop = None
      amod = None
      nsubj = None
      for out_edge in out_edges:
        next_node = out_edge[1]
        next_node_type = out_edge[2]['rel']
        if next_node_type in STOP_TYPES:
          # limit to one
          if (nsubj is None or not next_node_type == 'nsubj') and (det is None or not next_node_type == 'det') and (case is None or not next_node_type == 'case') and (mark is None or not next_node_type == 'mark') and (aux is None or not next_node_type == 'aux') and (neg is None or not next_node_type == 'neg') and (cop is None or not next_node_type == 'cop') and (amod is None or not next_node_type == 'amod'):
            if next_node_type == 'det':
              det = next_node
            elif next_node_type == 'case':
              case = next_node
            elif next_node_type == 'mark':
              mark = next_node
            elif next_node_type == 'nsubj':
              nsubj = next_node
            elif next_node_type == 'aux':
              aux = next_node
            elif next_node_type == 'neg':
              neg = next_node
            elif next_node_type == 'cop':
              cop = next_node
            elif next_node_type == 'amod':
              amod = next_node
            else:
              left_edges.append((next_node, next_node_type))
        elif next_node_type in ZAG_TYPE:
          if rel_chain[-1] == next_node_type:
            left_edges.append((next_node, next_node_type))
          else:
            right_edges.append((next_node, next_node_type))
        elif cur_rel in LEFT_NON_TERMINAL:
          left_non_terminal_edges.append((next_node, next_node_type))
        elif next_node_type == 'cc':
          cc = next_node
        elif next_node_type == 'conj':
          conj_edge = (next_node, next_node_type)
        elif next_node_type == 'compound':
          compound_edges.append((next_node, next_node_type))
        elif next_node_type == 'ccomp':
          ccomp = next_node
        else:
          right_edges.append((next_node, next_node_type))
      if not ccomp is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, ccomp, 'ccomp', sents, score)
      #left non terminal edges
      for (next_node, next_node_type) in left_non_terminal_edges:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, next_node, next_node_type, sents, score)
      if not nsubj is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, nsubj, 'nsubj', sents, score)
      #exclusive or: mark, aux
      if not mark is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, mark, 'mark', sents, score)
      elif not aux is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, aux, 'aux', sents, score)
      if not neg is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, neg, 'neg', sents, score)
      if not cop is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, cop, 'cop', sents, score)
      if not case is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, case, 'case', sents, score)
      if not det is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, det, 'det', sents, score)
      if not amod is None:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, amod, 'amod', sents, score)
      #left edges
      for (next_node, next_node_type) in left_edges:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, next_node, next_node_type, sents, score)
      #compounds
      for (next_node, next_node_type) in compound_edges:
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, next_node, next_node_type, sents, score)
      #current node
      left_sent.append(cur_node)
      
      #right edges
      for (next_node, next_node_type) in right_edges:

        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, next_node, next_node_type, sents, score)
      #conjunctions
      if cc and conj_edge:
        left_sent.append(cc)
        self.__find_sentences(G, nsubj_roots, rel_chain, word_chain, visited_nodes, left_sent, cur_node, conj_edge[0], conj_edge[1], sents, score)
      
      
      #prevent bigram repetition
      for i in xrange(0, len(left_sent) - 2):
        a = left_sent[i]
        b = left_sent[i + 1]
        if len(left_sent) >= 4:
          ca = left_sent.count(a)
          cb = left_sent.count(b)
          if ca > 1 and cb > 1:
            occ1 = [i for i,val in enumerate(left_sent) if val==a]
            occ2 = [i for i,val in enumerate(left_sent) if val==b]
            if occ1[-1] - occ1[-2] == occ2[-1] - occ2[-2]:
              # repeating 2+gram detected
              return False
            if len(occ2) > 2 and occ1[-1] - occ1[-2] == occ2[-2] - occ2[-3]:
              return False
            if len(occ1) > 2 and occ1[-2] - occ1[-3] == occ2[-1] - occ2[-2]:
              return False
      
      #prevent more than two occurences of a word
      if self.__exceeds_occ_limit(cur_node, visited_nodes, 2):
        return False
      else:
        if cur_node in visited_nodes:
          c = visited_nodes[cur_node] + 1
        else:
          c = 1
        visited_nodes[cur_node] = c
        
      if not left_sent[-1] is float and (cur_rel == 'dobj' or cur_rel == 'nsubj' or (out_degree == 0 and cur_rel not in STOP_TYPES and cur_rel not in LEFT_NON_TERMINAL)):
        #sent.append(','.join(rel_chain))
        sent = copy.deepcopy(left_sent)
        sent.append(score[0] / ((len(sent)) * self.min_word_score * self.min_score))
        if len(sents) == 0 or not abs(sents[-1][-1] - sent[-1]) <= 1e-6 :
          sents.append(sent)
        

  def __exceeds_occ_limit(self, cur_node, visited_nodes, limit):
    return cur_node in visited_nodes and visited_nodes[cur_node] > limit


  def create_scoring(self):
    self.scoring = {}
    min_score = 1
    for n in self.ranked_nodes:
      score = self.ranked_nodes[n]
      if score < min_score:
        min_score = score
      triple = json.loads(n)
      key = (triple[0][0], triple[1], triple[2][0])
      self.scoring[key] = score
    self.min_score = min_score


  def __is_hashtag(self, word):
    return '#' in word

  def __clean_word(self, word):
    return wn.morphy(word) or word

  def __triples2graph(self, triples):
    '''
    converts list of triples to 
    '''
    G = nx.MultiDiGraph()
    added_words = set()
    for triple in triples:
      words = [triple[x][0] for x in (0,2)]
      base_words = [self.__clean_word(w) for w in words]
      poss = [triple[x][1] for x in (0,2)]
      rel = triple[1]
      for i in xrange(0, 1):
        (word, base, pos) = (words[i], base_words[i], poss[i])
        if base not in added_words:
          added_words.add(base)
          pair = (word, pos)
          l = list()
          l.append(pair)
          G.add_node(base, word=l)
        else:
          for n in G.nodes(data=True):
            if n[0] == base:
              pair = (word, pos)
              n[1]['word'].append(pair)
      if not G.has_edge(base_words[0], base_words[1]):
        G.add_edge(base_words[0], base_words[1], rel=rel)
        
    return G

  def __parse_deps2trigram(self, nodes, depo, dept, dep):
    '''
    recursively parse dependencies into trigram model.
    '''
    if not depo is None and not dept is None and not dep is None:
      first = (self.__clean_word(depo[1]['word']), depo[0])
      second = (self.__clean_word(dept[1]['word']), dept[0])
      third = (self.__clean_word(dep[1]['word']), dep[0])
      self.dependency_tool.train_trigram(first, second, third)
    if 'deps' in dep[1]:
      subdeps = dep[1]['deps']
      for subdep in subdeps:
        node = nodes[subdeps[subdep][0]]
        next = (subdep, node)
        self.__parse_deps2trigram(nodes, dept, dep, next)
        
        

  def __parse_documents(self, documents):
    '''
    parses a list of documents using the stanford dependency parser.
    -> generates dependencytrigram model.
    -> generates dependency frequency model.
    -> generates morphologically-lossy graph of documents' union
    pickles results for future use.
    '''
    all_triples = list()
    if not os.path.isfile('../state/parsed.pickle') or not os.path.isfile('../state/graph.pickle'):
      sentiment = 0;
      for document in documents:
        sentiment = sentiment + TextBlob(document).sentiment.polarity;
        sents = nltk.sent_tokenize(document)
        print('Parsing document...')
        a = datetime.now()
        results = self.dep_parser.raw_parse_sents(sents)
        b = datetime.now()
        c = b - a
        print('[Parse Time] ' + str(c))
        self.roots = list()
        for result in results:
          for graph in result:
            triples = graph.triples()
            all_triples.extend(triples)
            self.roots.append(self.__clean_word(graph.root['word']))
            self.__parse_deps2trigram(graph.nodes, None, None, ('root', graph.root))

      self.overall_sentiment = sentiment / len(documents);
      print('Parsing dependencies...')     
      self.dependency_tool.put_all(all_triples)
      print('Generating union dependency graph...') 
      self.G = self.__triples2graph(all_triples)
      print('Pickling parse...')
      pickle.dump(self.G, open('../state/graph.pickle', 'wb'))
      pickle.dump(self.dependency_tool, open('../state/parsed.pickle', 'wb'))
      print('Parse pickled.')
    else:
      print('Unpickling parsed dependencies...')
      self.dependency_tool = pickle.load(open('../state/parsed.pickle', 'rb'))
      self.G = pickle.load(open('../state/graph.pickle', 'rb'))
      print('Parse unpickled.')

def main():
  documents = list()
  alien_document = open('corpus.txt').read().replace('\r\n', ' ').replace(',', '').lower()
  documents.append(alien_document)

  #with open('trump_tweets.csv') as f:
    #documents.append(f.read())

  #convert all to lower case
  for i, document in enumerate(documents):
    documents[i] = document.lower().replace('\r\n', ' ').replace(',', '')

  if not os.path.isfile('../state/resume.pickle'):
    ranker = multi_document_dependency_ranker()
    ranker.rank(documents)
    print('Dumping pickle...')
    pickle.dump(ranker, open('../state/resume.pickle', 'wb'))
  else:
    print('Loading pickle...')
    ranker = pickle.load(open('../state/resume.pickle', 'rb'))
    print('Loaded.')
    ranker.create_scoring()
    ranker.score_words()
    ranker.analyze()
  

if __name__ == '__main__':
  main()