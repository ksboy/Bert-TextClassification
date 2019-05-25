# coding=utf-8

import argparse
import os

def get_args():

    parser = argparse.ArgumentParser(description='BERT Baseline')


    #CNN 参数
    parser.add_argument("--filter_num", default=200,
                        type=int, help="filter 的数量")
    parser.add_argument("--filter_sizes", default="= 2 3 5",
                        type=str, help="filter 的 size")
    
    config = parser.parse_args()

    return config
