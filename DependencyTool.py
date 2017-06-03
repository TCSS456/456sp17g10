from nltk.corpus import stopwords;
import itertools;

class DependencyTool:
  def __init__(self):
    self.total = {'unfiltered': 0, 'filtered': 0};
    self.counts = {'unfiltered': {}, 'filtered': {}};
    self.stop = stopwords.words('english');

  def _put(self, dependency, group_name):
    self.total[group_name] = self.total[group_name] + 1;
    group = self.counts[group_name];
    if dependency in group:
      group[dependency] = group[dependency] + 1;
    else:
      group[dependency] = 1;
  
  def put(self, dependency):
    self._put(dependency, 'unfiltered');
    if not self.contains_stopwords(dependency):
      self._put(dependency, 'filtered');
  
  def put_all(self, dependencies):
    for dependency in dependencies:
      self.put(dependency);

  def dependencies(self, group_name):
    return self.counts[group_name].keys();

  def count(self, dependency, group_name):
    group = self.counts[group_name];
    if dependency in group:
      return group[dependency];
    else:
      return 0;
  
  # add-one smoothing
  def frequency(self, dependency, group_name='filtered'):
    group = self.counts[group_name];
    total = float(self.total[group_name]);
    if dependency in group:
      return (group[dependency] + 1) / total;
    else:
      return 1 / total;

  def contains_stopwords(self, dependency):
    if dependency[0][0] in self.stop or dependency[2][0] in self.stop:
      return True;
    else:
      return False;
  
  def dependency_frequencies(self, group_name):
    freq = {};
    for dependency in self.counts:
      freq[dependency] = self.frequency(dependency, group_name);
    return freq;

  def edge_frequencies(self, group_name):
    group = self.counts[group_name];
    total = float(self.total[group_name]);
    freq = {};
    for comb in itertools.combinations(group.keys(), 2):
      (first, second) = comb;
      if first[0] in second or first[2] in second:
        if comb in freq:
          freq[comb] = freq[comb] + 1;
        else:
          freq[comb] = 1;
      else:
        freq[comb] = 0;
    return freq;