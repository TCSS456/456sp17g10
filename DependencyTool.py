from nltk.corpus import stopwords;
import itertools;

class DependencyTool:
  def __init__(self, filter_stopwords=True):
    self.total = 0;
    self.counts = {};
    self.filter_stopwords = filter_stopwords;
    self.stop = stopwords.words('english');
  
  def put(self, dependency):
    if (self.filter_stopwords and not self.contains_stopwords(dependency)):
      self.total = self.total + 1;
      if dependency in self.counts:
        self.counts[dependency] = self.counts[dependency] + 1;
      else:
        self.counts[dependency] = 1;
  
  def put_all(self, dependencies):
    for dependency in dependencies:
      self.put(dependency);

  def dependencies(self):
    return self.counts.keys();

  def count(self, dependency):
    if dependency in self.counts:
      return self.counts[dependency];
    else:
      return 0;
  
  # add-one smoothing
  def frequency(self, dependency):
    if dependency in self.counts:
      return (self.counts[dependency] + 1) / float(self.total);
    else:
      return 1 / float(self.total);

  def contains_stopwords(self, dependency):
    if dependency[0][0] in self.stop or dependency[2][0] in self.stop:
      return True;
    else:
      return False;
  
  def dependency_frequencies(self):
    freq = {};
    for dependency in self.counts:
      freq[dependency] = self.frequency(dependency);
    return freq;

  def edge_frequencies(self):
    freq = {};
    for comb in itertools.combinations(self.counts.keys(), 2):
      (first, second) = comb;
      if first[0] in second or first[2] in second:
        if comb in freq:
          freq[comb] = freq[comb] + 1;
        else:
          freq[comb] = 1;
      else:
        freq[comb] = 0;
    return freq;