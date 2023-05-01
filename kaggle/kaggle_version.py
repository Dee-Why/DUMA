import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup  # , AdamW
from transformers import logging
from torch.nn import CrossEntropyLoss, MultiheadAttention
from torch.optim import AdamW
import numpy as np
import datetime
import random
import json
import csv
import re


def separate_seq2(sequence_output, flat_input_ids):
    qa_seq_output = sequence_output.new(sequence_output.size()).zero_()
    qa_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                         device=sequence_output.device,
                         dtype=torch.bool)
    p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                        device=sequence_output.device,
                        dtype=torch.bool)
    for i in range(flat_input_ids.size(0)):  # 这个是input
        sep_lst = []
        for idx, e in enumerate(flat_input_ids[i]):
            if e == 2:
                sep_lst.append(idx)
        assert len(sep_lst) == 2
        qa_seq_output[i, :sep_lst[0] - 1] = sequence_output[i, 1:sep_lst[0]]
        qa_mask[i, :sep_lst[0] - 1] = 0
        p_seq_output[i, :sep_lst[1] - sep_lst[0] -
                         1] = sequence_output[i, sep_lst[0] + 1: sep_lst[1]]
        p_mask[i, :sep_lst[1] - sep_lst[0] - 1] = 0
    return qa_seq_output, p_seq_output, qa_mask, p_mask


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DUMALayer(nn.Module):
    def __init__(self, d_model_size, num_heads):
        super(DUMALayer, self).__init__()
        self.attn_qa = MultiheadAttention(d_model_size, num_heads)
        self.attn_p = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None):
        qa_seq_representation = qa_seq_representation.permute(
            [1, 0, 2])  # attention需要的输入尺寸为 sequence length, batch_size, d_model size
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation, key=qa_seq_representation, query=p_seq_representation, key_padding_mask=qa_mask
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation, key=p_seq_representation, query=qa_seq_representation, key_padding_mask=p_mask
        )
        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])


class DUMA(nn.Module):
    def __init__(self, config, model_path, num_labels=5):
        super(DUMA, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_path, config=self.config)
        self.bert.gradient_checkpointing_enable()
        self.duma = DUMALayer(d_model_size=self.config.hidden_size,
                              num_heads=self.config.num_attention_heads)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.num_labels = num_labels

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_output = outputs.last_hidden_state
        qa_seq_output, p_seq_output, qa_mask, p_mask = separate_seq2(
            last_output, input_ids)
        enc_output_qa, enc_output_p = self.duma(  # DUMALayer
            qa_seq_output, p_seq_output, qa_mask, p_mask)
        fused_output = torch.cat([enc_output_qa, enc_output_p], dim=1)
        """
        pooled_output = torch.mean(fused_output, dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(pooled_output))
            else:
                logits += self.classifier(dropout(pooled_output))
        logits = logits / len(self.dropouts)
        """
        pooled_output = self.pooler(fused_output)
        droped_output = self.dropout(pooled_output)
        logits = self.classifier(droped_output)
        # print("logits.shape:",logits.shape, "self.num_labels:", self.num_labels) # torch[160,1]
        reshaped_logits = F.softmax(logits.view(-1, self.num_labels), dim=1)
        # print(reshaped_logits.shape) # torch.Size([32, 5])
        # print(reshaped_logits) # 每一行的和为0，并且非负，是概率分布
        return reshaped_logits


class DUMABert():
    def __init__(self, train_path, validation_path, vocab_path, model_path, wiki_path,
                 device, gpu, choices, max_len, train_batch_size, test_batch_size,
                 learning_rate, epsilon, epoches, save_model_path, random_seed, config_path=None):

        self.device = device
        self.gpu = gpu
        if self.gpu:
            seed = random_seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        self.wiki_dicts = json.load(open(wiki_path, 'r', encoding='UTF-8'))
        self.num_labels = choices
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epoches = epoches
        self.Tokenizer = BertTokenizer.from_pretrained(
            model_path, cache_dir='./Albert', num_choices=5)
        config = BertConfig.from_pretrained(model_path, num_choices=5)
        self.config = config
        # BertModel.from_pretrained(model_path,config=self.config)
        self.model = DUMA(config=config, model_path=model_path)
        self.train_data = self.load_data(train_path)
        self.validation_data = self.load_data(validation_path, Test=True)
        self.save_model_path = save_model_path

    # 读取csv文件，从csv文件中加载文本数据
    def read_file(self, path):
        # 只能处理

        text_list = []
        labels = []
        csv_file = open(path, encoding='UTF-8')
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)
        file_lines = csv.reader(csv_file)  # 文件内容的每一行
        if has_header:
            next(file_lines)
        for line in file_lines:
            label = int(line[6])
            text = str(line[0])
            re_match = re.match(r'(.*)（(.*)）', text)
            passage = re_match.group(1)
            hint = str(re_match.group(2)) + '?'
            wiki_choices = []
            for j in range(self.num_labels):
                choice_text = str(line[j + 1])  # 文本选项
                choice_wiki = self.wiki_dicts[choice_text]  # 将选项定向到wiki文本解释
                wiki_choices.append(hint + choice_text)
            # 将问题重复num_labels次
            content = [passage for i in range(self.num_labels)]
            pairs = (content, wiki_choices)
            text_list.append(pairs)
            labels.append(label)
        return text_list, labels

    def encode_fn(self, text_list, labels):
        input_ids, token_type_ids, attention_mask = [], [], []
        for text in text_list:
            encode_tokenizer = self.Tokenizer(text[1], text_pair=text[0], padding='max_length',
                                              truncation=True,
                                              max_length=self.max_len,
                                              return_tensors='pt')  # 搞不懂这个text_pair有什么作用？
            input_ids.append(encode_tokenizer['input_ids'].tolist())
            token_type_ids.append(encode_tokenizer['token_type_ids'].tolist())
            attention_mask.append(encode_tokenizer['attention_mask'].tolist())
        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        # print(input_ids[3].data)#测试用
        return TensorDataset(input_ids, token_type_ids, attention_mask, labels)

    # 加载训练数据or测试数据

    def load_data(self, path, Test=None):
        text_list, labels = self.read_file(path)
        Data = DataLoader(self.encode_fn(text_list, labels),
                          batch_size=self.train_batch_size if not Test else self.test_batch_size,
                          shuffle=False if Test else True, num_workers=2)  # 处理成多个batch的形式
        return Data

    def train_model(self):
        if self.gpu:
            self.model.to(self.device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters(
        )), lr=self.learning_rate, eps=self.epsilon)
        # filter(lambda p: p.requires_grad, self.model.parameters())
        epoches = self.epoches
        trainData = self.train_data
        testData = self.validation_data
        total_steps = len(trainData) * epoches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_Func = nn.CrossEntropyLoss()
        t0 = datetime.datetime.now()
        print('Train-----------')
        print(f'Every epoch have {len(trainData)} steps.')
        for epoch in range(epoches):
            self.model.train()
            train_loss = 0.0
            test_loss = 0.0
            test_accuracy = 0.0
            train_accuracy = 0.0
            print('Epoch: ', epoch + 1)
            for step, batch in enumerate(trainData):
                self.model.zero_grad()

                # print(batch.shape)

                input_ids = batch[0].view(-1, batch[0].size(-1))
                attention_mask = batch[1].view(-1, batch[1].size(-1))
                token_type_ids = batch[2].view(-1, batch[2].size(-1))

                labels = batch[3].to(self.device)
                logits = self.model(input_ids=input_ids.to(self.device),  # DUMA
                                    attention_mask=attention_mask.to(self.device),
                                    token_type_ids=token_type_ids.to(self.device),
                                    )

                loss = loss_Func(logits, labels)
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)  # 避免过拟合？
                optimizer.step()
                scheduler.step()

                logits = logits.detach()
                every_train_accuracy = self.model_accuracy(logits, labels)
                train_accuracy += every_train_accuracy
                if step % 50 == 0 and step > 0:
                    print('step:', step)
                    print(f'Accuracy: {train_accuracy / (step + 1):.4f}')
            t1 = datetime.datetime.now()
            print(f'Up to Epoch{epoch + 1} Time: {t1 - t0}')
            avg_train_loss = train_loss / len(trainData)
            print('Train loss: ', avg_train_loss)
            print('Train acc: ', train_accuracy / len(trainData))

            self.model.eval()
            for k, test_batch in enumerate(testData):
                with torch.no_grad():
                    input_ids = test_batch[0].view(-1, test_batch[0].size(-1))
                    attention_mask = test_batch[1].view(-1, test_batch[1].size(-1))
                    token_type_ids = test_batch[2].view(-1, test_batch[1].size(-1))
                    labels = test_batch[3].to(self.device)
                    logits = self.model(input_ids=input_ids.to(self.device),
                                        attention_mask=attention_mask.to(self.device),
                                        token_type_ids=token_type_ids.to(self.device),
                                        )
                    loss = loss_Func(logits, labels)
                    test_loss += loss.item()
                    logits = logits.detach()
                    test_accuracy += self.model_accuracy(logits, labels)

            avg_test_loss = test_loss / len(testData)
            avg_test_acc = test_accuracy / len(testData)
            print('Test--------------')
            print('Test loss: ', avg_test_loss)
            print('Test acc: ', avg_test_acc)
            if epoch == 0:
                Epoch_avg_test_acc = avg_test_acc
            """
            if avg_test_acc>0.5:
                #保存模型
                self.save_model(epoch)
            """

        print('训练结束！')
        t2 = datetime.datetime.now()
        print(f'Total time: {t2 - t0}')
        return Epoch_avg_test_acc

    def test_model(self):
        if self.gpu:
            self.model.to(self.device)
        testData = self.validation_data
        loss_Func = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_accuracy = 0.0
        self.model.eval()
        for k, test_batch in enumerate(testData):
            with torch.no_grad():
                input_ids = test_batch[0].view(-1, test_batch[0].size(-1))
                attention_mask = test_batch[1].view(-1, test_batch[1].size(-1))
                token_type_ids = test_batch[2].view(-1, test_batch[1].size(-1))
                labels = test_batch[3].to(self.device)
                logits = self.model(input_ids=input_ids.to(self.device),
                                    attention_mask=attention_mask.to(self.device),
                                    token_type_ids=token_type_ids.to(self.device),
                                    )
                loss = loss_Func(logits, labels)
                test_loss += loss.item()
                logits = logits.detach()
                test_accuracy += self.model_accuracy(logits, labels)

        avg_test_loss = test_loss / len(testData)
        avg_test_acc = test_accuracy / len(testData)
        print('Test--------------')
        print('Test loss: ', avg_test_loss)
        print('Test acc: ', avg_test_acc)

    def save_model(self, index):
        self.model.bert.save_pretrained(self.save_model_path + str(index) + '/' + 'bert.sav')
        self.Tokenizer.save_pretrained(self.save_model_path + str(index) + '/' + 'tokenizer.sav')
        torch.save(self.model, self.save_model_path + str(index) + '/' + 'model.sav')
        # model.save_pretrained(FIlE_PATH+'/Bert_Model/'+'-'+str(epoch))
        # tokenizer.save_pretrained((FIlE_PATH+'/Bert_Model/'+'-'+str(epoch)))

    def load_model(self, index):
        self.model = torch.load(self.save_model_path + str(index) + '/' + 'model.sav')
        self.model.bert = BertModel.from_pretrained(self.save_model_path + str(index) + '/' + 'bert.sav')
        self.Tokenizer = BertTokenizer.from_pretrained(self.save_model_path + str(index) + '/' + 'tokenizer.sav')

    def val_model(self):
        pass

    def model_accuracy(self, logits, labels):
        eq_logits = torch.eq(torch.max(logits, dim=1)[
                                 1], labels.flatten()).float()
        acc = eq_logits.sum().item() / len(eq_logits)
        return acc

    def test_accuracy(self, logits, labels, input_ids, Error_File):
        predict_labels = torch.max(logits, dim=1)[1]
        acc_sum = 0.
        for i in range(len(predict_labels)):
            if predict_labels[i] == labels[i]:
                acc_sum += 1.
            else:
                print(str(predict_labels[i]) + '  ' + str(
                    self.Tokenizer.convert_ids_to_tokens(input_ids[i]) + '\n'), file=Error_File)


if __name__ == '__main__':
    logging.set_verbosity_error()  # 只保留报错信息，而无warning信息
    continue_training_on_index = None  # 这个是设定继续在那个版本的基础上继续训练
    save_index = 18  # 本次训练结果的版本号
    epochs = 6  # 一次 train 的 epoch数量
    learning_rates = [3e-5, 3e-6, 1e-6]
    seed = [11, ]
    model_file = ''
    trained_model_file = './Bert-base-trained_model'
    model_name = '/Bert-RobertA/'
    data_file = '/kaggle/input/chineseriddle/'

    # Error_File_Name='./Error-Results.txt'
    # Error_File=open(Error_File_Name,'w')
    # 加载GPU
    gpu = False
    if torch.cuda.is_available():
        gpu = True
    device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')

    tri_avg_test_acc = 0.0
    bert_model = DUMABert(
        train_path=data_file + 'train.csv',
        validation_path=data_file + 'val.csv',
        config_path=None,  # model_file+'/bert_config.json',
        vocab_path='nghuyong/ernie-1.0',  # model_file+'/vocab.txt'
        model_path='nghuyong/ernie-1.0',  # model_file+'/bert_model.bin'
        wiki_path=data_file + 'wiki_info_v3.json',
        device=device,
        gpu=gpu,
        choices=5,
        max_len=256,
        train_batch_size=16,
        test_batch_size=16,
        learning_rate=learning_rates[0],
        epsilon=1e-8,
        epoches=epochs,
        random_seed=seed[0],
        save_model_path=trained_model_file + model_name,
    )

    torch.cuda.empty_cache()
    # 加载上一次训练的结果数据
    if continue_training_on_index is not None:
        print('loading model ' + str(continue_training_on_index))
        bert_model.load_model(continue_training_on_index)
        bert_model.model.bert.gradient_checkpointing_enable()

    # 训练过程
    print('train_batch_size ', bert_model.train_batch_size)
    print('learning_rate    ', bert_model.learning_rate)
    print('seed ', seed[0])
    bert_model.train_model()
    print('-' * 20, 'weight decay', '-' * 20)
    bert_model.epoches = 6
    bert_model.learning_rate = learning_rates[1]
    bert_model.train_model()
    print('-' * 20, 'weight decay', '-' * 20)
    bert_model.learning_ratelea = learning_rates[2]
    bert_model.train_model()

    # 保存训练结果
    bert_model.save_model_path = './Bert-base-trained_model/Bert-RobertA/'
    bert_model.model.bert.save_pretrained(bert_model.save_model_path + str(save_index) + '/' + 'bert.sav')
    bert_model.Tokenizer.save_pretrained(bert_model.save_model_path + str(save_index) + '/' + 'tokenizer.sav')
    torch.save(bert_model.model, bert_model.save_model_path + str(save_index) + '/' + 'model.sav')