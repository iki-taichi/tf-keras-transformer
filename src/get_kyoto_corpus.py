# coding: utf-8


"""
Download Japanese-English Bilingual Corpus of Wikipedia's Kyoto Articles
And make it csv file for transformer fitting

You can read documents about the license of the corpus at the following url: 
https://alaginrc.nict.go.jp/WikiCorpus/index_E.html#license 
"""


import argparse

import os
from urllib.request import urlretrieve
import tarfile

import csv
import glob
import xml.etree.ElementTree as ET

import numpy as np


DOWNLOAD_URL='https://alaginrc.nict.go.jp/WikiCorpus/cgi-bin/dl1.cgi'
TARGET_FILE_NAME = 'wiki_corpus_2.01.tar.gz'


def download_data(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    file_path = os.path.join(work_dir, TARGET_FILE_NAME)
    urlretrieve(DOWNLOAD_URL, file_path)

    tar = tarfile.open(file_path, 'r:gz') 
    tar.extractall(work_dir) 
    tar.close()


def by_sentence(input_path, output_path, valid_ratio):
    target_path = os.path.join(input_path, '*/*.xml')
    xmls = glob.glob(target_path, recursive=True)
    print('to convert %d xml files...'%(len(xmls)))
    
    pairs = []

    for xml in xmls:
        try:
            root = ET.parse(xml).getroot()
        except Exception as e:
            print('%s skipped because of %s'%(xml, e))
            continue
        
        for sen in root.findall('.//sen'):
            ja = sen.find('j')
            en = sen.find('e[@type=\'check\']')

            if ja is not None and en is not None:
                pairs.append([en.text, ja.text])
            
            if len(pairs) < 5:
                print('data sample:(%s)%s'%(xml, pairs[-1]))
    
    created_files = []
    
    if valid_ratio == 0:
        with open(output_path, 'w') as f:
            csv.writer(f).writerows(pairs)
            created_files.append(output_path)
    else:
        output_path_prefix, ext = os.path.splitext(output_path)
        output_path_valid = output_path_prefix + '_valid' + ext
        valid_len = int(len(pairs)*valid_ratio)
        
        np.random.shuffle(pairs)
        
        with open(output_path_valid, 'w') as f:
            csv.writer(f).writerows(pairs[:valid_len])
            created_files.append(output_path_valid)
        
        with open(output_path, 'w') as f:
            csv.writer(f).writerows(pairs[valid_len:])
            created_files.append(output_path)
    
    return created_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make csv files.')
    parser.add_argument(
            '-w', '--work_dir', type=str, default='data/kyoto_corpus',
            help='path to working dir where downloaded data is expanded'
        )
    parser.add_argument(
            '-o', '--output_path', type=str, default='data/kyoto_en_ja.csv',
            help='path to output file'
        )
    parser.add_argument(
            '-v', '--valid_ratio', type=float, default=0.1,
            help='ratio of rows for validation'
        )
    
    args = parser.parse_args()
    
    print('start downloading...')
    download_data(args.work_dir)
    print('downloaded')
    
    print('start editing files...')
    created_files = by_sentence(args.work_dir, args.output_path, args.valid_ratio)
    print('%s were created'%(','.join(created_files)))
    