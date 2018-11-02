#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path

import dill
import numpy as np

import phsic
import phsic.config
import phsic.encoder
import phsic.kernel
import phsic.phsic

"""
# naming rule

## raw text
- {data_dir}/{data_basename}
- e.g., data/nmt/train.en

## feature vector
- {out_dir}/feat/{data_basename}.{encoder_config}.vec
- e.g., work/nmt/feat/train.en.NormalizedUSE

## model
- {out_dir}/model/{data_basename_X}.{kernel_config_X}-{encoder_config_X}.{data_basename_Y}.{kernel_config_Y}-{encoder_config_Y}
- e.g., work/nmt/model/train.en.Linear-NormalizedUSE.train.ja.Linear-NormalizedBov-FasttextJa(limit100000)
## score
- {out_dir}/score/{data_basename_X}.{kernel_config_X}-{encoder_config_X}.{data_basename_Y}.{kernel_config_Y}-{encoder_config_Y}[.{X_test}.{Y_test}].phsic
#     - e.g., work/nmt/model/train.en.Linear-NormalizedUSE.train.ja.Linear-NormalizedBov-FasttextJa(limit100000).test.en.test.ja.phsic
"""


def prepare_feature_vecs(path_to_data,
                         path_to_feat_vec,
                         encoder, encoder_config, limit_words, emb_file):
    # load
    if os.path.isfile(path_to_feat_vec):
        print('Loading feature vectors: {}'.format(path_to_feat_vec),
              file=sys.stderr)
        X = dill.load(open(path_to_feat_vec, 'rb'))

    # or encode
    else:
        if encoder[0] is None:
            encoder[0] = phsic.encoder.encoder(encoder_config,
                                               limit_words=limit_words,
                                               emb_file=emb_file)
        X = encoder[0].encode_batch_from_file(path_to_data)
        print('Dumping feature fectors: {}'.format(path_to_feat_vec),
              file=sys.stderr)
        p = Path(path_to_feat_vec)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(path_to_feat_vec, 'wb') as f:
            dill.dump(X, f)

    return X


def run(args):
    # modify configs of kernels and encoders
    # e.g., (Cos, Bov FasttextEn) -> (Linear, NormalizedBov FasttextEn)
    kernel_config1, encoder_config1 = phsic.config.convert_config_for_phsic(
        args.kernel1, args.encoder1)
    kernel_config2, encoder_config2 = phsic.config.convert_config_for_phsic(
        args.kernel2, args.encoder2)
    print('kernel_config1:  {}'.format(kernel_config1), file=sys.stderr)
    print('encoder_config1: {}'.format(encoder_config1), file=sys.stderr)
    print('kernel_config2:  {}'.format(kernel_config2), file=sys.stderr)
    print('encoder_config2: {}'.format(encoder_config2), file=sys.stderr)

    # .
    X_train_basename = os.path.basename(args.file_X)
    Y_train_basename = os.path.basename(args.file_Y)
    X_train_feat_path = os.path.join(args.out_dir, 'feat',
                                     '{data_basename}.{encoder_config}.vec'.format(
                                         data_basename=X_train_basename,
                                         encoder_config='-'.join(
                                             encoder_config1)))
    Y_train_feat_path = os.path.join(args.out_dir, 'feat',
                                     '{data_basename}.{encoder_config}.vec'.format(
                                         data_basename=Y_train_basename,
                                         encoder_config='-'.join(
                                             encoder_config2)))

    if args.X_test and args.Y_test:
        X_test_basename = os.path.basename(args.X_test)
        Y_test_basename = os.path.basename(args.Y_test)
        X_test_feat_path = os.path.join(args.out_dir, 'feat',
                                        '{data_basename}.{encoder_config}.vec'.format(
                                            data_basename=X_test_basename,
                                            encoder_config='-'.join(
                                                encoder_config1)))
        Y_test_feat_path = os.path.join(args.out_dir, 'feat',
                                        '{data_basename}.{encoder_config}.vec'.format(
                                            data_basename=Y_test_basename,
                                            encoder_config='-'.join(
                                                encoder_config2)))

    # encoders
    ## encoder1 = None として prepare_feature_vecs() が新しい encoder を返すようにすると, obj1 <- encoder1; obj1 <- encoder2 の参照を維持できない
    encoder1 = [None]
    if encoder_config1 == encoder_config2 and args.limit_words1 == args.limit_words2:
        encoder2 = encoder1
    else:
        encoder2 = [None]

    # PREPROCESSING: load feature vectors or encode raw texts into feature vectors
    X = prepare_feature_vecs(args.file_X,
                             X_train_feat_path,
                             encoder1, encoder_config1, args.limit_words1,
                             args.emb1)
    Y = prepare_feature_vecs(args.file_Y,
                             Y_train_feat_path,
                             encoder2, encoder_config2, args.limit_words2,
                             args.emb2)

    # kernel function
    # (repr, repr) -> float
    kernel1, kernel_batch1 = phsic.kernel.positive_definite_kernel(
        kernel_config1, X)
    kernel2, kernel_batch2 = phsic.kernel.positive_definite_kernel(
        kernel_config2, Y)

    # fit
    kernel_type1 = kernel_config1[0]
    kernel_type2 = kernel_config2[0]
    if kernel_type1 == 'Linear' and kernel_type2 == 'Linear':
        model = phsic.phsic.PHSIC()
        model.fit_XY(X, Y)
    else:
        model = phsic.phsic.PHSIC_ICD(args.no_centering)
        model.fit_XY(
            X, Y,
            kernel1, kernel2,
            args.dim1, args.dim2,
            k_batch=kernel_batch1, l_batch=kernel_batch2)

    params_text = '{kernel1}-{encoder1}.{kernel2}-{encoder2}{dim1}{dim2}'.format(
        kernel1='-'.join(args.kernel1),
        encoder1='-'.join(args.encoder1),
        kernel2='-'.join(args.kernel2),
        encoder2='-'.join(args.encoder2),
        dim1='' if kernel_type1 == 'Linear' and kernel_type2 == 'Linear' else '.{}'.format(
            args.dim1),
        dim2='' if kernel_type1 == 'Linear' and kernel_type2 == 'Linear' else '.{}'.format(
            args.dim2),
    )

    # predict
    outpath = '{prefix}.{params}.phsic'.format(
        prefix=args.out_prefix,
        params=params_text,
    )
    if args.X_test and args.Y_test:
        X_test = prepare_feature_vecs(args.X_test,
                                      X_test_feat_path,
                                      encoder1, encoder_config1,
                                      args.limit_words1, args.emb1)
        Y_test = prepare_feature_vecs(args.Y_test,
                                      Y_test_feat_path,
                                      encoder2, encoder_config2,
                                      args.limit_words2, args.emb2)

        with open(outpath, 'w') as fo:
            phsics = model.predict_batch_XY(X_test, Y_test)
            print('writting output file...', file=sys.stderr)
            np.savetxt(fo, phsics)
    else:
        with open(outpath, 'w') as fo:
            phsics = model.predict_batch_training_data()
            print('writting output file...', file=sys.stderr)
            np.savetxt(fo, phsics)
    print('written: {}'.format(outpath), file=sys.stderr)

    print('Done!', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        prog='PHSIC',
        description='Give phsic scores to paird data.')
    parser.add_argument('file_X')
    parser.add_argument('file_Y')
    ## kernel 1
    parser.add_argument('--kernel1', nargs='+', help='e.g. Gaussian 1.0',
                        required=True)
    parser.add_argument('--encoder1', nargs='+', help='e.g., SumBov FasttextEn',
                        required=True)
    parser.add_argument('--emb1', type=str, required=True)
    ## kernel 2
    parser.add_argument('--kernel2', nargs='+', help='e.g. Gaussian 1.0')
    parser.add_argument('--encoder2', nargs='+', help='e.g., SumBov FasttextEn')
    parser.add_argument('--emb2', type=str, required=True)

    parser.add_argument('--dim1', type=int, required=True)
    parser.add_argument('--dim2', type=int, required=True)

    parser.add_argument('--limit_words1', type=int)
    parser.add_argument('--limit_words2', type=int)

    parser.add_argument('--no_centering', action='store_true')  # default: False

    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--out_prefix', required=True)

    parser.add_argument('--X_test', '--test_file_X')
    parser.add_argument('--Y_test', '--test_file_Y')

    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(phsic.__version__))

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2), file=sys.stderr)

    run(args)


if __name__ == '__main__':
    main()
