from bs4 import BeautifulSoup
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import unidecode
from word2number import w2n
from pycontractions import Contractions
import gensim.downloader as api
from gensim import corpora, models, similarities
from collections import defaultdict
import math


class NLP():
    nlp = None
    doc = None
    model = None

    def __init__(self, spacy_model='en_core_web_sm', gensim_model='glove-twitter-25'):
        self.nlp = spacy.load(spacy_model)
        self.model = api.load(gensim_model)
        self.cont = Contractions(kv_model=self.model)

    def remove_html(self, text):
        """Strip HTML tags from text"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=" ")

    def remove_accents(self, text):
        """Remove accented characters from text for non-english words"""
        return unidecode.unidecode(text)

    def expand_contractions(self, text):
        """Convert contractions into whole words. e.g. can't -> can not"""
        return list(self.cont.expand_texts([text], precise=True))[0]

    def preprocess(self, text, remove_numbers=False, remove_stopwords=False, excluded_sw=None, toke=False):
        """Preprocess using standard protocols. 
        @param remove_numbers converts words to digits and removes
        @param remove_stopwords removes stop words
        @param excluded_sw is any stopwords to exclude
        @param toke if true, return tokens, default return text 
        """
        text = self.remove_html(text)
        text = self.remove_accents(text)
        text = self.expand_contractions(text)

        if toke or remove_numbers or remove_stopwords:
            if excluded_sw is not None:
                for w in excluded_sw:
                    self.nlp.vocab[w].is_stop = False
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if token.pos_ == 'NUM' and not remove_numbers:
                    tokens.append(w2n.word_to_num(token.text))
                elif not token.is_stop:
                    tokens.append(token.text)
            if toke:
                return tokens
            text = " ".join(tokens)
        return text

    def lemmatize(self, tokens, toke=False):

        lookups = Lookups()
        lookups.add_table('lemma_index', lemma_index)
        lookups.add_table('lemma_exc', lemma_exc)
        lookups.add_table('lemma_rules', lemma_rules)
        lemmatizer = Lemmatizer(lookups)

        lemmas = []
        for t in tokens:
            lemmas.append(lemmatizer(token.text, token.tag_))

        if toke:
            return lemmas

        return " ".join(lemmas)

    def get_syllables(self, word):
        count = 0
        vowels = ("a", "e", "i", "o", "u", "y")
        prev = False
        for c in word:
            vowel = c in vowels
            if vowel and not prev:
                count += 1
            prev = vowel
        return count

    def get_lexical_density(self, tokens):
        c_words = t_words = 0

        cont_pos = ['PROPN', 'NOUN', 'VERB', 'ADJ', 'ADV']
        for t in tokens:
            if token.pos_ in cont_pos:
                c_words += 1
                t_words += 1
            elif token.pos_ != 'PUNCT':
                t_words += 1

        return round((c_words / t_words), 4)

    def get_coherence(self, text):
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents if len(sent) >= 2]
        frequency = defaultdict(int)
        token_sents = []
        for s in sentences:
            tmp = []
            for t in self.preprocess(s, remove_stopwords=True, excluded_sw=['no', 'not'], toke=True):
                tmp.append(t.text)
                frequency[t] += 1
            token_sents.append(tmp)

        vocab = [[word for word in sent if frequency[word] > 1]
                 for sent in token_sents]
        dictionary = corpora.Dictionary(vocab)
        corpus = [dictionary.doc2bow(word) for word in vocab]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
        corpus_lsi = lsi[corpus_tfidf]

        sums = {}
        topic_count = max([len(line) for line in corpus_lsi])
        for line in corpus_lsi:
            for topic in line:
                t_num = topic[0]
                if t_num not in sums:
                    sums[t_num] = abs(topic[1])
                else:
                    sums[t_num] += abs(topic(1))
        best_topic = max(zip(sums.values(), sums.keys()))[1]
        ordered = []
        i = 0
        for line in corpus_lsi:
            ordered.append((i, line[topic][1]))
            i += 1

        ordered = sorted(ordered, key=lambda x: x[1], reverse=True)
        threshold = ordered[0][1] - (0.90 * (ordered[0][1] - ordered[-1][1]))
        problem_sentences = []
        for s in ordered:
            if s[1] < threshold:
                problem_sentences.append((s[1]), s)
        problem_sentences = [s for s in ordered if s[1] < threshold]

        output = {}
        for p in problem_sentences:
            output[p[0]] = (p[1], str(sentences[p[0]]))

        return output

    def get_readability(self, text):
        scores = {}

        doc = self.nlp(text)
        sentence _count = len(doc)

        words = self.preprocess(text, toke=True)
        characters = 0
        for word in words:
            characters += len(word)
        word_count = len(words)
        
        syllable_count = 0
        complex_words = 0
        for word in words:
            c = self.get_syllables(word)
            syllable_count += c
            if c >= 3 and not word[0].isupper():
                complex_words += 1
        avgwps = word_count / sentence_count

        # Automated Readability Index
        ari = 0.0
        ari_grade = 0
        if word_count > 0:
            ari = 4.71 * (characters / word_count) + 0.5 * \
                (word_count / sentence_count) - 21.43
        if ari < 2:
            ari_grade = 0
        elif ari > 12:
            ari_grade = 13
        else:
            ari_grade = ari
        scores["ari"] = (ari, ari_grade)

        # Flesch Reading Ease
        flesch_reading_ease = 101
        fre_grade = 0
        if word_count > 0 and sentence_count > 0:
            flesch_reading_ease = 206.835 - \
                1.015(word_count / sentence_count) - \
                84.6(syllable_count / word_count)
        if flesch_reading_ease > 100:
            fre_grade = 4
        elif flesch_reading_ease > 90.0:
            fre_grade = 5
        elif flesch_reading_ease > 80.0:
            fre_grade = 6
        elif flesch_reading_ease > 70.0:
            fre_grade = 7
        elif flesch_reading_ease > 60.0:
            fre_grade = 9
        elif flesch_reading_ease > 50:
            fre_grade = 12
        else:
            fre_grade = 13
        scores["flesch_reading_ease"] = (flesch_reading_ease, fre_grade)

        # Flesch-Kincaid Grade Level
        fkg = 0.0
        if word_count > 0 and sentence_count > 0:
            fkg = 0.39(word_count / sentence_count) + \
                11.8(syllable_count / word_count) - 15.59
        scores["flesch_kinkaid_grade_level"] = (fkg, int(fkg))

        # Gunning Fog Index
        gfi = 0.0
        gfi_grade = 0
        if sentence_count > 0 and word_count > 0:
            gfi = 0.4 * ((word_count / sentence_count) +
                        100(complex_words / word_count))
        if gfi < 6:
            gfi_grade = 5
        elif gfi <= 12:
            gfi_grade = int(gfi)
        else:
            gfi_grade = 13
        scores["gunning_fog_index"] = (gfi, gfi_grade)

        # SMOG Readability
        smog = 0.0
        smog_grade = 0
        if sentence_count > 0:
            smog = 1.0430 * math.sqrt(complex_words *
                                    (30 / sentence_count)) + 3.1291
        if smog >= 13:
            smog_grade = 13
        else:
            smog_grade = int(smog)
        scores["smog_readability"] = (smog, smog_grade)

        # ColemanLiauIndex
        coleman = 0.0
        coleman_grade = 0
        if word_count > 0:
            coleman = (5.89 * (characters / word_count)) - \
                (30 * (sentence_count / word_count)) - 15.8
        if coleman >= 13:
            coleman_grade = 13
        else:
            coleman_grade = int(coleman)
        scores["coleman_liau"] = (coleman, coleman_grade)

        # LIX & RIX
        lix = 0.0
        rix = 0.0
        lix_grade = 0
        rix_grade = 0
        if sentence_count > 0 and word_count > 0:
            long_words = 0
            for word in words:
                if len(word) >= 7:
                    long_words += 1
            lix = word_count / sentence_count + ((100. * long_words) / word_count)
            rix = long_words / sentence_count
        if lix >= 13:
            lix_grade = 13
        else:
            lix_grade = int(lix)
        if rix >= 13:
            rix_grade = 13
        else:
            rix_grade = int(rix)
        scores["LIX"] = (lix, lix_grade)
        scores["RIX"] = (rix, rix_grade)

        count = 0
        avg = 0.0
        for k, v in scores.items:
            avg += v[1]
            count += 1
        scores["AVERAGE_GRADE"] = (avg / count, int(avg / count))

        return scores
