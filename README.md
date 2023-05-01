# Project: Solving Riddles with Transformer Model: DUMA-Bert

#### Team Members: Mingchao Liu, Jialin Guo, Jingyi Hao

environment:

* python: 3.9.12
* pytorch: 1.13.1
* transformers: 4.26.1

>  to run the code, you may need to reset the file path for reading training data


**`separate_seq2`** function: this function is used for seperating sequence. It takes in one sequence tensor and a input token ID and seperates them into four tensors `qa_seq_output`, `p_seq_output`,` qa_mask`,` p_mask`. To do so, it seperares the input sequence through `[SEP]` token to get `qa_seq_output` that represents question and answer and `p_seq_output` that represents passage. Then, it creates two new boolen value ` qa_mask`,` p_mask` that represents maks. `qa_seq_output` is get by taking hidden states from the bigining of input sequence till `[SEP]` and not include `[CLS]` and `p_seq_output` contains the rest of sequence. With these four ouputs, it is ready to be passed on to `DUMA` class.



**`BertPooler`** class: In this class, we extract a pooled representation from our input sequence and stroed it for future tasks. To be specific, the `__init__` fuction contains a linear layer that takes in the hidden states of BERT and its activation is tangent function. In `forward`, the inputs tensor of hidden sates is of shape [batch _size, sequance_length, hidden_size]. Due to BERT's bidiretinoality, first content of token is [CLS] token that conatins all necessary information. Hence, we extract the the first token in each sequence and pass that through linear layer and activation function to get the desired `pooled_output` tensor, that of shape [batch_size, hidden_size].



**`DUMALayer`** class: The imputs of this layer are two sets of sequence representations, `qa_seq_representation` and `p_seq_representation`. The layer first applies a multi-head attention mechanism (`MultiheadAttention`) to compute the attention scores between `qa_seq_representation` and `p_seq_representation`, and similarly for `p_seq_representation` and `qa_seq_representation`. The `key_padding_mask` parameters are used to mask certain tokens in the input sequences during the attention calculation. The output of the layer. a tuple of two tensors, `enc_output_qa` and `enc_output_p`, is the attention output for the `qa_seq_representation` and `p_seq_representation` respectively, which are tensors that are permuted to their original shape of `batch_size, sequence_length, d_model_size` .



**`DUMA`** class : The model combines a pre-trained BERT model with a DUMA layer and a linear classifier for fine-tuning on a downstream task. The `BertModel` is loaded from a pre-trained checkpoint, and the `gradient_checkpointing_enable()` method can help with memory efficiency during training. The `DUMALayer` is defined with `d_model_size` (`hidden_size`) and `num_heads `(`num_attention_heads`) and  values from the BERT configuration respectively. The `DUMALayer` takes the output sequence representations from BERT for both the question-answer (`qa_seq_representation`) and passage (`p_seq_representation`) inputs. The `BertPooler` can pool the output sequence representation from the last layer of BERT into a fixed-size representation, which is then passed through a `dropout` layer with a dropout probability of 0.5. Finally, a linear classifier with one output unit is defined with `nn.Linear` to predict a binary classification label (e.g. positive/negative) for the input passage and question-answer pair.



**`DUMABert`** class: in this class, we make use of all the previous class, define the needed paramaters, write out functions we need and prepare to start training our model. In `__init__` function, we define all the parameters, for eample, random seed, gpu etc. In `read_file` function, we read in the datasets the consist of a passage, a question and five answer choices and stored the last column as `labels` that contains the correct answers. We pairs up the answer choices with wiki hints to get `wiki_choices`. Since for each question, we have five choices.  So we combine the passage and question as `content` and conacatnate it with `wiki_choices` and repeat it five times to get five paris and append to `text_list`. In `encode_fn` function, we convert the texts to tokens using BERT tokenizer and store them to a tensor dataset that conatins `input_ids`, `token_type_ids`, `attention_mask`, `labels`. Then, we use `load_data` function to load data into torch dataloader. Finally, in `train_model`, `model_accuracy` and `test_accuracy` function, we initialize paramaters for  train model on training set and test its performances on tets set and metric to record the evaluation.



**Training cell**: in this last cell, we define all necessaey infromation like batch size, learning rate, seed, epoches and train our DUMA-Bert model and record its performance.

