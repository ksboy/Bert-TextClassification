# coding=utf-8
from main import main
import os

if __name__ == "__main__":

    model_name = "BertCNN"
    data_dir = "../yelp_review_polarity_csv_splited/"
    output_dir = "./output"
    cache_dir = "./cache"
    log_dir = "./log"

    model_times = "model_1/"  # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "../bert-base-uncased/vocab.txt"
    bert_model_dir = "../bert-base-uncased"

    do_train = True
    do_test = True

    # map(lambda: x, y: os.path.join(x, y),

    from Processors.Yelp2Processor import Yelp2Processor

    if model_name == "BertOrigin":
        from BertOrigin import args

        main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertCNN":
        from BertCNN import args

        main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertATT":
        from BertATT import args

        main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertRCNN":
        from BertRCNN import args

        main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertRNNCNN":
        from BertRNNCNN import args

        main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir,
                           do_train, do_test),
             model_times, Yelp2Processor)
