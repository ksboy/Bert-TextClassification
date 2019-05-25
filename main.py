# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from Utils.utils import get_device
from Utils.Classifier_utils import load_data

from Utils.train_evalute import train, evaluate


def main(config,model_times, myProcessor, model_config = None):
    # print(os.path.join(config.output_dir, model_times))
    if not os.path.exists(os.path.join(config.output_dir , model_times)):
        os.makedirs(os.path.join(config.output_dir , model_times))

    if not os.path.exists(os.path.join(config.cache_dir , model_times)):
        os.makedirs(os.path.join(config.cache_dir , model_times))

    output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)  # 模型输出文件
    output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]

    device, n_gpu = get_device(gpu_ids[0])  # 设备准备

    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    """ 设定随机种子 """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    """ 数据准备 """
    processor = myProcessor()  # 整个文件的代码只需要改此处即可
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择

    label_list = processor.get_labels()
    num_labels = len(label_list)


    if True:

        train_dataloader, train_examples_len = load_data(
            config.data_dir, tokenizer, processor, config.max_seq_length, config.train_batch_size, "train")
        dev_dataloader, _ = load_data(
            config.data_dir, tokenizer, processor, config.max_seq_length, config.dev_batch_size, "dev")

        num_train_optimization_steps = int(
            train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs
        
        """ 模型准备 """
        print("model name is {}".format(config.model_name))
        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in model_config.filter_sizes.split()]
            model = BertCNN.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels, 
                n_filters=model_config.filter_num, filter_sizes=filter_sizes)
        elif config.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertRNNCNN":
            from BertRCNN.BertRNNCNN import BertRNNCNN
            model = BertRNNCNN.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model,device_ids=gpu_ids)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer,
              criterion, config.gradient_accumulation_steps, device, label_list, output_model_file, output_config_file, config.log_dir, config.eval_every, config.early_stop)

    if True:
        """ Test """
        test_dataloader, _ = load_data(
            config.data_dir, tokenizer, processor, config.max_seq_length, config.test_batch_size, "test")

        bert_config = BertConfig(output_config_file)
        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin(bert_config, num_labels=num_labels)
        elif config.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in config.filter_sizes.split()]
            model = BertCNN(bert_config, num_labels=num_labels,
                            n_filters=config.filter_num, filter_sizes=filter_sizes)
        elif config.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT.from_pretrained(
                config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertRCNN":
                from BertRCNN.BertRCNN import BertRCNN
                model = BertRCNN.from_pretrained(
                    config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertRNNCNN":
                from BertRCNN.BertRNNCNN import BertRNNCNN
                model = BertRNNCNN.from_pretrained(
                    config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)

        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # test the model
        test_loss, test_acc, test_report, test_auc = evaluate(
            model, test_dataloader, criterion, device, label_list)
        print("-------------- Test -------------")
        print('\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | AUC:{test_auc}')

        for label in label_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
        print_list = ['macro avg', 'weighted avg']

        for label in print_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
