import sys

import gensim
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import tensorflow_hub as hub

import phsic.utils

# --------------------------------
# EONCODER_CONFIG
# --------------------------------

"""
- `ENCODER_CONFIG`
    - `[ENCODER_TYPE]`  \# sentence encoder
    - `[ENCODER_TYPE, WORDVEC_KEY]`  \# w/word vector

- `ENCODER_TYPE`
    - w/word vector
        - `'SumBov'`: sum of bag-of-vectors in given sentence
        - `'AveBov'`: average of bag-of-vectors in given sentence
        - `'NormalizedSumBov'`: normalized `'SumBov'`
    - sentence encoder
        - `'USE'`: universal sentence encoder w/DAN
        - `'NormalizedUSE'`: normalized `'USE'`

- `WORDVEC_KEY`
    - `'FasttextCs'`
    - `'FasttextDe'`
    - `'FasttextJa'`
    - `'FasttextEn'`
    - `'FasttextVi'`
    - `'FasttextCsBin'`
    - `'FasttextDeBin'`
    - `'FasttextJaBin'`
    - `'FasttextViBin'`
"""


def encoder(*args, **kwargs):
    return Encoder(*args, **kwargs)


class Encoder:

    def __init__(self,
                 encoder_config,
                 limit_words=None,
                 eps=1e-8,
                 emb_file=None):

        self.eps = eps
        self.encoder_type = encoder_config[0]

        # load word vector file
        if self.encoder_type in {'SumBov', 'AveBov', 'NormalizedSumBov'}:
            wordvec_key = encoder_config[1]
            self.wv_model = load_wv_from_wordvec_key(
                wordvec_key, limit_words=limit_words, emb_file=emb_file)
            # self.sentence_encoder = lambda self, sentence: sum_of_wordvecs(self.wv_model)(sentence)
        # load sentence encoder
        elif self.encoder_type in {'USE', 'NormalizedUSE'}:
            tf_module_url = 'https://tfhub.dev/google/universal-sentence-encoder/1'
            print('Loading...: {}'.format(tf_module_url), file=sys.stderr)
            self.sentence_encoder = hub.Module(tf_module_url)

    def encode(self, sentence):
        # return self.sentence_encoder(sentence)
        if isinstance(sentence, (list, tuple)):
            sentence = ' '.join(sentence)
        if self.encoder_type == 'SumBov':
            return self.sum_of_wordvecs(self.wv_model)(sentence)
        elif self.encoder_type == 'AveBov':
            return self.ave_of_wordvecs(self.wv_model)(sentence)
        elif self.encoder_type == 'NormalizedSumBov':
            return self.normalized_sum_of_wordvecs(self.wv_model)(sentence)

    def __call__(self, sentence):
        return self.encode(sentence)

    def encode_batch_from_file(self, path):
        print('Encoding sentences in {}'.format(path), file=sys.stderr)
        return self.encode_batch(open(path))

    def encode_batch(self, sentences):
        """
        sentences: iter of sentences

        Return
        ----------
        iter of sentence embeddings
        """
        if self.encoder_type in {'SumBov', 'AveBov', 'NormalizedSumBov'}:
            return [self.encode(s) for s in sentences]
        elif self.encoder_type == 'USE':
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(),
                             tf.tables_initializer()])
                return session.run(self.sentence_encoder(list(sentences)))
        elif self.encoder_type == 'NormalizedUSE':
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(),
                             tf.tables_initializer()])
                return sklearn.preprocessing.normalize(
                    session.run(self.sentence_encoder(list(sentences))),
                    norm='l2')

    def sum_of_wordvecs(self, wv_model):
        """
        wv_model -> sentence -> vec
        """
        wv_type = type(wv_model)

        def _sum_of_wordvecs(sentence):
            vecs = [_to_vec(word.rstrip(",."), wv_model, wv_type) for word in
                    sentence.split() if
                    _in_vocab(word.rstrip(",."), wv_model, wv_type)]
            if vecs:
                return sum(vecs)
            else:
                return phsic.utils.sample_from_unit_hypersphere(
                    wv_model.vector_size)
                # return np.zeros((wv_model.vector_size,)) + self.eps

        return _sum_of_wordvecs

    def ave_of_wordvecs(self, wv_model):
        """
        wv_model -> sentence -> vec
        """
        wv_type = type(wv_model)

        def _sum_of_wordvecs(sentence):
            vecs = [_to_vec(word.rstrip(",.，．、。"), wv_model, wv_type) for word
                    in sentence.split() if
                    _in_vocab(word.rstrip(",.，．、。"), wv_model, wv_type)]
            if vecs:
                return np.average(vecs, axis=0)
            else:
                return phsic.utils.sample_from_unit_hypersphere(
                    wv_model.vector_size)
                # return np.zeros((wv_model.vector_size,)) + self.eps

        return _sum_of_wordvecs

    def normalized_sum_of_wordvecs(self, wv_model):
        """
        wv_model -> sentence -> vec
        """
        wv_type = type(wv_model)

        def _normalized_sum_of_wordvecs(sentence):
            vecs = [_to_vec(word.rstrip(",."), wv_model, wv_type) for word in
                    sentence.split() if
                    _in_vocab(word.rstrip(",."), wv_model, wv_type)]
            if vecs:
                vec = sum(vecs)
            else:
                vec = np.zeros((wv_model.vector_size,)) + self.eps
            norm = np.linalg.norm(vec) + self.eps
            return vec / norm

        return _normalized_sum_of_wordvecs


# --------------------------------
# word vector
# --------------------------------

"""
- `WORDVEC_FORMAT`
    - `'word2vec.txt'`
    - `'word2vec.bin'`
    - `'fasttext.vec'`
    - `'fasttext.bin'`


- `WV_MODEL`
    - gensim word2vec/fasttext model


- `WV_MODEL_TYPE`
    - `TYPE_WORD2VEC`
    - `TYPE_FASTTEXT`
"""

TYPE_WORD2VEC = gensim.models.keyedvectors.Word2VecKeyedVectors
TYPE_FASTTEXT = gensim.models.fasttext.FastText

# WORDVEC_KEY: dict(relpath, WORDVEC_FORMAT)
WORDVEC_DICT = {
    # fasttext.bin
    'Word2vecEnPrivate': dict(
        relpath='../data/wordvecs/word2vec/private.en.bin',
        wordvec_format='word2vec.bin',
    ),
    'Word2vecJaPrivate': dict(
        relpath='../data/wordvecs/word2vec/private.ja.bin',
        wordvec_format='word2vec.bin',
    ),
    # fasttext.vec
    'FasttextCs': dict(
        relpath='../data/wordvecs/fasttext/cc.cs.300.vec',
        wordvec_format='fasttext.vec',
    ),
    'FasttextDe': dict(
        relpath='../data/wordvecs/fasttext/cc.de.300.vec',
        wordvec_format='fasttext.vec',
    ),
    'FasttextJa': dict(
        relpath='../data/wordvecs/fasttext/cc.ja.300.vec',
        wordvec_format='fasttext.vec',
    ),
    'FasttextEn': dict(
        relpath='../data/wordvecs/fasttext/crawl-300d-2M.vec',
        wordvec_format='fasttext.vec',
    ),
    'FasttextEnLight': dict(
        relpath='../data/wordvecs/fasttext/crawl-300d-2M.vec.light',
        wordvec_format='fasttext.vec',
    ),
    'FasttextVi': dict(
        relpath='../data/wordvecs/fasttext/cc.vi.300.vec',
        wordvec_format='fasttext.vec',
    ),
    # fasttext.bin
    'FasttextCsBin': dict(
        relpath='../data/wordvecs/fasttext/cc.cs.300.bin',
        wordvec_format='fasttext.bin',
    ),
    'FasttextDeBin': dict(
        relpath='../data/wordvecs/fasttext/cc.de.300.bin',
        wordvec_format='fasttext.bin',
    ),
    'FasttextJaBin': dict(
        relpath='../data/wordvecs/fasttext/cc.ja.300.bin',
        wordvec_format='fasttext.bin',
    ),
    'FasttextViBin': dict(
        relpath='../data/wordvecs/fasttext/cc.vi.300.bin',
        wordvec_format='fasttext.bin',
    ),
}


# load model

# limit_words works when wv_model_type is TYPE_WORD2VEC,
# that is, wordvec_format is 'fasttext.vec'

def load_wv_from_wordvec_key(wordvec_key, limit_words=None, emb_file=None):
    # limit_words works when wv_model_type is TYPE_WORD2VEC,
    # that is, wordvec_format is 'fasttext.vec'
    wordvec_format = WORDVEC_DICT[wordvec_key]['wordvec_format']
    return load_wordvec(emb_file,
                        limit_words=limit_words,
                        wordvec_format=wordvec_format)


def load_wordvec(abspath_wordvec, limit_words=None, wordvec_format='word2vec'):
    # ref. https://github.com/RaRe-Technologies/gensim/issues/814
    print('Loading...: {} (limit words: {})'.format(
        abspath_wordvec, limit_words), file=sys.stderr)

    if wordvec_format == 'fasttext.vec':
        return gensim.models.KeyedVectors.load_word2vec_format(abspath_wordvec,
                                                               binary=False,
                                                               limit=limit_words)
    elif wordvec_format == 'fasttext.bin':
        return gensim.models.fasttext.FastText.load_fasttext_format(
            abspath_wordvec)
    elif wordvec_format == 'word2vec.bin':
        return gensim.models.KeyedVectors.load_word2vec_format(abspath_wordvec,
                                                               binary=True)


# utils

def _to_vec(word, wv_model, wv_model_type=TYPE_WORD2VEC):
    if wv_model_type == TYPE_WORD2VEC:
        return wv_model[word]
    elif wv_model_type == TYPE_FASTTEXT:
        return wv_model.wv[word]


def _in_vocab(word, wv_model, wv_model_type=TYPE_WORD2VEC):
    if wv_model_type == TYPE_WORD2VEC:
        return word in wv_model.vocab  # tests if word present in vocab
    elif wv_model_type == TYPE_FASTTEXT:
        return word in wv_model  # tests if vector present for word

# sentence representation

# def sum_of_wordvecs(wv_model):
#     """
#     wv_model -> sentence -> vec
#     """

#     wv_model_type = type(wv_model)

#     def _sum_of_wordvecs(sentence):
#         return sum(_to_vec(word.rstrip(",."), wv_model, wv_model_type) for word in sentence.split()
#                    if _in_vocab(word.rstrip(",."), wv_model, wv_model_type))

#     return _sum_of_wordvecs
