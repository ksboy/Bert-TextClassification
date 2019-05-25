# coding=utf-8
from main import main


if __name__ == "__main__":

    model_name = "BertOrigin"
    data_dir = "/search/hadoop02/suanfa/songyingxin/data/SST-2"
    output_dir = ".sst_output/" 
    cache_dir = ".sst_cache"
    log_dir = ".sst_log/" 

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"
    

    from Processors.SST2Processor import Sst2Processor

    if model_name == "BertOrigin":
        import args

        main(
            args.get_args(data_dir, output_dir, cache_dir,
                          bert_vocab_file, bert_model_dir, log_dir), model_times, Sst2Processor)
    
    elif model_name == "BertCNN":
        from BertCNN import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir), 
                           model_times, Sst2Processor)
    elif model_name == "BertATT":
        from BertATT import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir), 
                           model_times, Sst2Processor)
    elif model_name == "BertRCNN":
        from BertRCNN import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir), 
                           model_times, Sst2Processor)
        

