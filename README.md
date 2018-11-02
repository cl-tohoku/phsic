# Pointwise Hilbert–Schmidt Independence Criterion (PHSIC)

Compute *co-occurrence* between two objects utilizing *similarities*.

For example, given consistent sentence pairs:

| X                                                            | Y                  |
| ------------------------------------------------------------ | ------------------ |
| They had breakfast at the hotel.                             | They are full now. |
| They had breakfast at ten.                                   | I'm full.          |
| She had breakfast with her friends.                          | She felt happy.    |
| They had breakfast with their friends at the Japanese restaurant. | They felt happy.   |
| He have trouble with his homework.                           | He cries.          |
| I have trouble associating with others.                      | I cry.             |

PHSIC can give high scores to consistent pairs in terms of the given pairs:

| X                                            | Y                     | score  |
| -------------------------------------------- | --------------------- | ------ |
| They had breakfast at the hotel.             | They are full now.    | 0.1134 |
| They had breakfast at an Italian restaurant. | They are stuffed now. | 0.0023 |
| I have dinner.                               | I have dinner again.  | 0.0023 |

## Installation

```
$ pip install phsic-cli
```

This will install `phsic` command to your environment:

```
$ phsic --help
```

## Basic Usage

Download pre-trained wordvecs (e.g. fasttext):

```
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
$ unzip crawl-300d-2M.vec.zip
```

Prepare dataset:

```
$ TAB="$(printf '\t')"
$ cat << EOF > train.txt
They had breakfast at the hotel.${TAB}They are full now.
They had breakfast at ten.${TAB}I'm full.
She had breakfast with her friends.${TAB}She felt happy.
They had breakfast with their friends at the Japanese restaurant.${TAB}They felt happy.
He have trouble with his homework.${TAB}He cries.
I have trouble associating with others.${TAB}I cry.
EOF
$ cut -f 1 train.txt > train_X.txt
$ cut -f 2 train.txt > train_Y.txt
$ cat << EOF > test.txt
They had breakfast at the hotel.${TAB}They are full now.
They had breakfast at an Italian restaurant.${TAB}They are stuffed now.
I have dinner.${TAB}I have dinner again.
EOF
$ cut -f 1 test.txt > test_X.txt
$ cut -f 2 test.txt > test_Y.txt
```

Then, train and predict:

```
$ phsic train_X.txt train_Y.txt --kernel1 Gaussian 1.0 --encoder1 SumBov FasttextEn --emb1 crawl-300d-2M.vec --kernel2 Gaussian 1.0 --encoder2 SumBov FasttextEn --emb2 crawl-300d-2M.vec --limit_words1 10000 --limit_words2 10000 --dim1 3 --dim2 3 --out_prefix toy --out_dir out --X_test test_X.txt --Y_test test_Y.txt
$ cat toy.Gaussian-1.0-SumBov-FasttextEn.Gaussian-1.0-SumBov-FasttextEn.3.3.phsic
1.134489336180434238e-01
2.320408776101631244e-03
2.321869174772554344e-03
```

## Citation

```
@InProceedings{D18-1203,
  author = 	"Yokoi, Sho
        and Kobayashi, Sosuke
        and Fukumizu, Kenji
        and Suzuki, Jun
        and Inui, Kentaro",
  title = 	"Pointwise HSIC: A Linear-Time Kernelized Co-occurrence Norm for Sparse Linguistic Expressions",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1763--1775",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1203"
}
```
