import os
import sys
import pickle
sys.path.append('../')
from multi_document_dependency_ranker import multi_document_dependency_ranker

documents = list()
alien_document = open('../corpus.txt').read().replace('\r\n', ' ').replace(',', '').lower()
documents.append(alien_document)

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