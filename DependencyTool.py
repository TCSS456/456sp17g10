from nltk.corpus import stopwords
import itertools

class DependencyTool:
  def __init__(self):
    self.total = {'unfiltered': 0, 'filtered': 0, 'lossy-unfiltered': 0}
    self.counts = {'unfiltered': {}, 'filtered': {}, 'lossy-unfiltered': {}}
    self.trigram_counts = {}
    self.bigram_counts = {}
    self.stop = stopwords.words('english')

  def _put(self, dependency, group_name):
    '''
    adds a dependency to a specific group
    '''
    self.total[group_name] = self.total[group_name] + 1
    group = self.counts[group_name]
    if dependency in group:
      group[dependency] = group[dependency] + 1
    else:
      group[dependency] = 1
  
  def put(self, dependency):
    '''
    adds a dependency
    '''
    self._put(dependency, 'unfiltered')
    self._put((dependency[0][0], dependency[1], dependency[2][0]), 'lossy-unfiltered')
    if not self.contains_stopwords(dependency):
      self._put(dependency, 'filtered')
  
  def put_all(self, dependencies):
    '''
    adds an iterable object of dependencies
    '''
    for dependency in dependencies:
      self.put(dependency)

  def dependencies(self, group_name):
    '''
    returns a list of dependencies within a group
    '''
    return self.counts[group_name].keys()

  def count(self, dependency, group_name):
    '''
    returns the number of occurences of a dependency within a group
    '''
    group = self.counts[group_name]
    if dependency in group:
      return group[dependency]
    else:
      return 0

  def train_trigram(self, first, second, third):
    '''
    adds a trigram to the training set for the trigram model
    format:
    each item = (morph_word, dependency_type)
    '''
    bi_key = (first, second)
    tri_key = (first, second, third)

    bi_count = 1
    tri_count = 1

    if tri_key in self.trigram_counts:
      tri_count = tri_count + self.trigram_counts[tri_key]
    self.trigram_counts[tri_key] = tri_count

    if bi_key in self.bigram_counts:
      bi_count = bi_count + self.bigram_counts[bi_key]
    self.bigram_counts[bi_key] = bi_count
  
  def trigram_probability(self, first, second, third):
    '''
    returns the probability of a trigram
    format:
    each item = (morph_word, dependency_type)
    '''
    bi_key = (first, second)
    tri_key = (first, second, third)

    if bi_key in self.bigram_counts and tri_key in self.trigram_counts:
      bi_count = self.bigram_counts[bi_key]
      tri_count = self.trigram_counts[tri_key]
      probability = tri_count / float(bi_count)
    else:
      probability = float(0)
    return probability

  def frequency(self, dependency, group_name='filtered'):
    '''
    returns the frequency of a certain dependency within a given group
    additive smoothing
    '''
    group = self.counts[group_name]
    total = float(self.total[group_name])
    if dependency in group:
      return (group[dependency] + 1) / total
    else:
      return 1 / total

  def contains_stopwords(self, dependency):
    '''
    returns boolean if the dependency contains stopwords
    '''
    if dependency[0][0] in self.stop or dependency[2][0] in self.stop:
      return True
    else:
      return False
  
  def dependency_frequencies(self, group_name):
    '''
    returns a map of all dependecncy frequencies within a given group
    '''
    freq = {}
    for dependency in self.counts:
      freq[dependency] = self.frequency(dependency, group_name)
    return freq