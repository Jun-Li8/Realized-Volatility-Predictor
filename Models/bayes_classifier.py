import math
import re

def f_score(actual, predict):
    if len(actual) != len(predict):
        print("Length of actual must equal to predict")
        return 0

    pos_pos = 0
    pos_neu = 0
    pos_neg = 0
    neu_pos = 0
    neu_neu = 0
    neu_neg = 0
    neg_pos = 0
    neg_neu = 0
    neg_neg = 0
    for i in range(0,len(predict)):
        if predict[i] == '1' and actual[i] == '1':
            neg_neg += 1
        elif predict[i] == '1' and actual[i] == '2':
            neu_neg += 1
        elif predict[i] == '1' and actual[i] == '3':
            pos_neg += 1
        elif predict[i] == '2' and actual[i] == '1':
            neg_neu += 1
        elif predict[i] == '2' and actual[i] == '2':
            neu_neu += 1
        elif predict[i] == '2' and actual[i] == '3':
            pos_neu += 1
        elif predict[i] == '3' and actual[i] == '1':
            neg_pos += 1
        elif predict[i] == '3' and actual[i] == '2':
            neu_pos += 1
        elif predict[i] == '3' and actual[i] == '3':
            pos_pos += 1
        else:
            pass

    precision = float(pos_pos + 1)/float(pos_pos + neg_pos + neu_pos + 1)
    recall = float(pos_pos + 1)/float(pos_pos + pos_neu + pos_neg + 1)
    f_score_pos = float(2.0)*precision*recall/(precision+recall)

    precision = float(neu_neu + 1)/float(pos_neu + neg_neu + neu_neu + 1)
    recall = float(neu_neu + 1)/float(neu_pos + neu_neu + neu_neg + 1)
    f_score_neu = float(2.0)*precision*recall/(precision+recall)

    precision = float(neg_neg + 1)/float(pos_neg + neg_neg + neu_neg + 1)
    recall = float(neg_neg + 1)/float(neg_pos + neg_neu + neg_neg + 1)
    f_score_neg = float(2.0)*precision*recall/(precision+recall)

    return f_score_pos, f_score_neg, f_score_neu


class Bayes_Classifier:

    def __init__(self):
        self.numpositive = 0
        self.numnegative = 0
        self.numneutral = 0
        self.wordfrequency = {}
        self.positivewords = []
        self.neutralwords = []
        self.negativewords = []
        self.totalwords = 0
        self.punctuations = ['!', '@', '#', '$', '%%', '^', '&', '*', '(', ')', '-', '_',
            '=', '+', '`', '~', ';', ',', '.', '<', '>', '/', ':', '?']
        self.stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again',
        'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
        'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
        'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am',
        'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until',
        'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
        'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
        'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
        'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in',
        'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
        'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
        'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which',
        'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
        'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

    def train(self, lines):
        for line in lines:
            line = line.replace('\n', '')
            for each in self.punctuations:
                line = line.replace(each, '')

            fields = line.split('|')
            rating = fields[0]
            text = fields[1].split(' ')
            for word in text:
                word = word.lower()

            if rating == '1':
                self.numnegative += 1
            elif rating == '2':
                self.numneutral += 1
            elif rating == '3':
                self.numpositive += 1
            else:
                print("Invalid Rating")
                raise

            for word in text:
                rep = rating + word

                if rep in self.wordfrequency:
                    self.wordfrequency[rep] += 1
                else:
                    self.wordfrequency[rep] = 1

                if rating == '1' and word not in self.negativewords:
                    self.negativewords.append(word)
                elif rating == '2' and word not in self.neutralwords:
                    self.neutralwords.append(word)
                else:
                    self.positivewords.append(word)

                self.totalwords += 1

    def classify(self, lines):
        actual = []
        predict = []
        for i, line in enumerate(lines):
            line = line.replace('\n', '')
            for each in self.punctuations:
                line = line.replace(each, '')

            fields = line.split('|')
            try:
                rating = fields[0]
                text = fields[1].split(' ')
            except:
                text = fields[0]
                rating = 0

            for word in text:
                word = word.lower()

            ppos = self.numpositive / (self.numpositive + self.numnegative + self.numneutral)
            pneg = self.numnegative / (self.numpositive + self.numnegative + self.numneutral)
            pneu = self.numneutral / (self.numpositive + self.numnegative + self.numneutral)

            for word in text:
                pos = '3' + word
                if pos in self.wordfrequency:
                    posprob = (self.wordfrequency[pos] + 1) / (self.numpositive + 1)
                else:
                    posprob = 1 / (self.numpositive + 1)

                neg = '1' + word
                if neg in self.wordfrequency:
                    negprob = (self.wordfrequency[neg] + 1) / (self.numnegative + 1)
                else:
                    negprob = 1 / (self.numnegative + 1)

                neu = '2' + word
                if neu in self.wordfrequency:
                    neuprob = (self.wordfrequency[neu] + 1) / (self.numneutral + 1)
                else:
                    neuprob = 1 / (self.numneutral + 1)

                ppos *= posprob
                pneg *= negprob
                pneu *= neuprob

            if ppos == 0 and pneg == 0 and pneu != 0:
                predict.append('2')
            elif ppos == 0 and pneu == 0 and pneg != 0:
                predict.append('1')
            elif pneg == 0 and pneu == 0 and ppos != 0:
                predict.append('3')
            else:
                try:
                    pos_new = math.log(ppos)
                    neg_new = math.log(pneg)
                    neu_new = math.log(pneu)
                    l = [pos_new, neg_new, neu_new]

                    i = l.index(max(l))
                    if i == 0:
                        predict.append('3')
                    elif i == 1:
                        predict.append('1')
                    else:
                        predict.append('2')

                    actual.append(rating)
                except:
                    print(word)
                    pass
                    #print(self.numpositive, self.numnegative, self.numneutral, i)
        return actual, predict
