import pickle
import string
from os import path
from html.parser import HTMLParser
from sklearn.feature_extraction import stop_words


class DataParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._last_tag = ''
        self._topics = True
        self._topics_list = []
        self._text = ''
        self._lewis_split = ''
        self.train_data = []
        self.test_data = []

    def handle_starttag(self, tag, attrs):
        if tag != 'D'.lower():
            self._last_tag = tag
        if tag == 'REUTERS'.lower():
            self._topics_list = []
            for attr in attrs:
                if attr[0] == 'TOPICS'.lower():
                    self._topics = (attr[1] == 'YES')
                elif attr[0] == 'LEWISSPLIT'.lower():
                    self._lewis_split = attr[1]

    def handle_endtag(self, tag):
        if tag == 'REUTERS'.lower() and self._topics:
            if self._lewis_split == 'TEST':
                self.test_data.append([self._text, self._topics_list])
            if self._lewis_split == 'TRAIN':
                self.train_data.append([self._text, self._topics_list])

    def handle_data(self, data):
        if self._last_tag == 'BODY'.lower() and self._topics:
            data = (' '.join(data.split())).strip()
            if data != '':
                self._text = data
        if self._last_tag == 'TOPICS'.lower() and self._topics:
            data = data.strip()
            if data != '':
                self._topics_list.append(data)


def extract_data():
    def remove_stopwords(s):
        s = s.lower().split()
        s = ' '.join([word for word in s if word not in (list(stop_words.ENGLISH_STOP_WORDS) + ['reuters'])])
        for p in string.punctuation:
            s = s.replace(p, '')
        return s

    # IF files exist THEN load ELSE generate & dump
    if path.isfile('train_data.pkl') and path.isfile('test_data.pkl'):
        print('Load train data')
        with open('train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        print('Load test data')
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
    else:
        parser = DataParser()
        for i in range(22):
            print('Handle SGM %03d' % i)
            with open('reuters21578/reut2-%03d' % i + '.sgm', encoding='latin-1') as f:
                data = f.read()
            parser.feed(data)
        train_data, test_data = parser.train_data, parser.test_data

        for i in range(len(train_data)):
            train_data[i][0] = remove_stopwords(train_data[i][0])
        for i in range(len(test_data)):
            test_data[i][0] = remove_stopwords(test_data[i][0])

        print('Save train data')
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        print('Save test data')
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)

    return train_data, test_data
