# coding=utf-8
from main import main
import args

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
        main(args.get_args(model_name, data_dir, output_dir, cache_dir, log_dir ,bert_vocab_file, bert_model_dir),

             model_times, Yelp2Processor)
    elif model_name == "BertCNN":
        from BertCNN import args_model

        main(args.get_args(model_name, data_dir, output_dir, cache_dir, log_dir, bert_vocab_file, bert_model_dir),
             model_times, Yelp2Processor, args_model.get_args())
    elif model_name == "BertATT":

        main(args.get_args(model_name, data_dir, output_dir, cache_dir, log_dir, bert_vocab_file, bert_model_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertRCNN":

        main(args.get_args(model_name, data_dir, output_dir, cache_dir, log_dir, bert_vocab_file, bert_model_dir),
             model_times, Yelp2Processor)
    elif model_name == "BertRNNCNN":

        main(args.get_args(model_name, data_dir, output_dir, cache_dir, log_dir, bert_vocab_file, bert_model_dir),
             model_times, Yelp2Processor)
