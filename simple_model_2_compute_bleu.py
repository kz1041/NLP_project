# Load data
# Preprocess data
# Create data loaders
# Batch implementation
# Design experiments
# Code for results and figures



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pandas as pd 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load in Data

# Create a pandas dataframe containing the sentence pairs
def create_dataframe(filename1, filename2, col_name1, col_name2):
    lines_1 = []
    with open(filename1, 'r') as f:
        line = f.readline()
        while line:
            lines_1.append(line.strip())
            line = f.readline()
    
    lines_2 = []
    with open(filename2, 'r') as f:
        line = f.readline()
        while line:
            lines_2.append(line.strip())
            line = f.readline()
    
    df = pd.DataFrame()
    df[col_name1] = lines_1
    df[col_name2] = lines_2
    
    # Remove empty lines from dataframe
    df = df[(df[col_name1] != '') & (df[col_name2] != '')]
    return df



def create_dataframe_length_group(filename1, filename2, col_name1, col_name2):
    lines_1 = []
    with open(filename1, 'r') as f:
        line = f.readline()
        while line:
            lines_1.append(line.strip())
            line = f.readline()
    
    lines_2 = []
    with open(filename2, 'r') as f:
        line = f.readline()
        while line:
            lines_2.append(line.strip())
            line = f.readline()
    
    df = pd.DataFrame()
    df[col_name1] = lines_1
    df[col_name2] = lines_2
    
    # Remove empty lines from dataframe
    df = df[(df[col_name1] != '') & (df[col_name2] != '')]

    # split by length group
    df = df[df.apply(lambda x: 50<= len(x[col_name1].split(' '))<10000, axis=1)]

    return df
# 12.1
# < 10: 17.4
# 10-15: 14.1
# 15-20: 11.8
# 20 - 30 :12.3
#>30: 9.9
# 30-50: 11
# >50: 6.3

# Load training set for Vietnamese-English
vi_en_train = create_dataframe('./iwslt-vi-en/train.tok.vi', './iwslt-vi-en/train.tok.en', 'vi', 'en')

# Load training set for Chinese-English
zh_en_train = create_dataframe('./iwslt-zh-en/train.tok.zh', './iwslt-zh-en/train.tok.en', 'zh', 'en')

# Load validation set for Vietnamese-English
vi_en_val = create_dataframe_length_group('./iwslt-vi-en/dev.tok.vi', './iwslt-vi-en/dev.tok.en', 'vi', 'en')

# Load validation set for Chinese-English
zh_en_val = create_dataframe('./iwslt-zh-en/dev.tok.zh', './iwslt-zh-en/dev.tok.en', 'zh', 'en')

# Load test set for Vietnamese-English
vi_en_test = create_dataframe_length_group('./iwslt-vi-en/test.tok.vi', './iwslt-vi-en/test.tok.en', 'vi', 'en')

# Load test set for Chinese-English
zh_en_test = create_dataframe('./iwslt-zh-en/test.tok.zh', './iwslt-zh-en/test.tok.en', 'zh', 'en')












# Preprocessing

from collections import Counter
import string

# Sentence lengths will actually be +1 because of EOS token
MAX_SENTENCE_LENGTH_ZH_EN = 100
MAX_SENTENCE_LENGTH_VI_EN = 100
MAX_LENGTH = 100
# Need to take into account 4 IDX values
MAX_VOCAB_SIZE = 6666

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Simple tokenization of dataset, splitting by space
def tokenize(line, language):
    """
    @param: line - line to split
    @param: language - language of line
    """
    # Replace &apos; and &quot
    if language == 'en':
        line = line.replace("&apos;", "'")
        line = line.replace("&quot;", '"')
        
    tokens = line.lower().split(' ') if language == 'en' else line.split(' ')
    tokens = [token for token in tokens if (token not in string.whitespace)]
    return tokens

# From lab 3
def tokenize_dataset(lines, language):
    """
    @param: lines - list of lines to split
    @param: language - language of lines
    """
    token_dataset = []
    all_tokens = []
    for line in lines:
        tokens = tokenize(line, language)
        token_dataset.append(tokens)
        all_tokens += tokens
        
    return token_dataset, all_tokens

# Tokenize training sets
vi_en_train_vi_tokens_dataset, vi_en_train_vi_all_tokens = tokenize_dataset(vi_en_train['vi'].tolist(), 'vi')
vi_en_train_en_tokens_dataset, vi_en_train_en_all_tokens = tokenize_dataset(vi_en_train['en'].tolist(), 'en')
zh_en_train_zh_tokens_dataset, zh_en_train_zh_all_tokens = tokenize_dataset(zh_en_train['zh'].tolist(), 'zh')
zh_en_train_en_tokens_dataset, zh_en_train_en_all_tokens = tokenize_dataset(zh_en_train['en'].tolist(), 'en')

# Tokenize validation sets
vi_en_val_vi_tokens_dataset, _ = tokenize_dataset(vi_en_val['vi'].tolist(), 'vi')
vi_en_val_en_tokens_dataset, _ = tokenize_dataset(vi_en_val['en'].tolist(), 'en')
zh_en_val_zh_tokens_dataset, _ = tokenize_dataset(zh_en_val['zh'].tolist(), 'zh')
zh_en_val_en_tokens_dataset, _ = tokenize_dataset(zh_en_val['en'].tolist(), 'en')

# Tokenize test sets
vi_en_test_vi_tokens_dataset, _ = tokenize_dataset(vi_en_test['vi'].tolist(), 'vi')
vi_en_test_en_tokens_dataset, _ = tokenize_dataset(vi_en_test['en'].tolist(), 'en')
zh_en_test_zh_tokens_dataset, _ = tokenize_dataset(zh_en_test['zh'].tolist(), 'zh')
zh_en_test_en_tokens_dataset, _ = tokenize_dataset(zh_en_test['en'].tolist(), 'en')








# From lab 3
def build_vocabulary(tokens, max_vocab_size):
    token_counter = Counter(tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(4,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>', '<SOS>', '<EOS>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    token2id['<SOS>'] = SOS_IDX
    token2id['<EOS>'] = EOS_IDX
    return token2id, id2token




# Chinese-to-English English id2token and token2id
zh_to_en_en_token2id, zh_to_en_en_id2token = build_vocabulary(zh_en_train_en_all_tokens, MAX_VOCAB_SIZE)

# Chinese-to-English Chinese id2token and token2id
zh_to_en_zh_token2id, zh_to_en_zh_id2token = build_vocabulary(zh_en_train_zh_all_tokens, MAX_VOCAB_SIZE)

# Vietnamese-to-English English id2token and token2id
vi_to_en_en_token2id, vi_to_en_en_id2token = build_vocabulary(vi_en_train_en_all_tokens, MAX_VOCAB_SIZE)

# Vietnamese-to-English Vietnamese id2token and token2id
vi_to_en_vi_token2id, vi_to_en_vi_id2token = build_vocabulary(vi_en_train_vi_all_tokens, MAX_VOCAB_SIZE)




# Token to index dataset
# From lab 3
def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

# Training indices
zh_to_en_en_train_indices = token2index_dataset(zh_en_train_en_tokens_dataset, zh_to_en_en_token2id)
zh_to_en_zh_train_indices = token2index_dataset(zh_en_train_zh_tokens_dataset, zh_to_en_zh_token2id)
vi_to_en_en_train_indices = token2index_dataset(vi_en_train_en_tokens_dataset, vi_to_en_en_token2id)
vi_to_en_vi_train_indices = token2index_dataset(vi_en_train_vi_tokens_dataset, vi_to_en_vi_token2id)

# Validation indices
zh_to_en_en_val_indices = token2index_dataset(zh_en_val_en_tokens_dataset, zh_to_en_en_token2id)
zh_to_en_zh_val_indices = token2index_dataset(zh_en_val_zh_tokens_dataset, zh_to_en_zh_token2id)
vi_to_en_en_val_indices = token2index_dataset(vi_en_val_en_tokens_dataset, vi_to_en_en_token2id)
vi_to_en_vi_val_indices = token2index_dataset(vi_en_val_vi_tokens_dataset, vi_to_en_vi_token2id)

# Test indices
zh_to_en_en_test_indices = token2index_dataset(zh_en_test_en_tokens_dataset, zh_to_en_en_token2id)
zh_to_en_zh_test_indices = token2index_dataset(zh_en_test_zh_tokens_dataset, zh_to_en_zh_token2id)
vi_to_en_en_test_indices = token2index_dataset(vi_en_test_en_tokens_dataset, vi_to_en_en_token2id)
vi_to_en_vi_test_indices = token2index_dataset(vi_en_test_vi_tokens_dataset, vi_to_en_vi_token2id)







# Create dataset
from torch.utils.data import Dataset

# From lab 3
class LanguagePairs(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
        
    def __init__(self, source_sentences, target_sentences, max_sentence_length):
        """
        @param source_sentences: list of source sentence tokens
        @param target_sentences: list of target sentence tokens 
        @param max_sentence_length: the maximum length of a sentence
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.max_sentence_length = max_sentence_length
        assert (len(self.source_sentences) == len(self.target_sentences))

    def __len__(self):
        return len(self.source_sentences)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        source_sentence_idx = self.source_sentences[key][:self.max_sentence_length]
        target_sentence_idx = self.target_sentences[key][:self.max_sentence_length]
        return [source_sentence_idx, target_sentence_idx, len(source_sentence_idx), len(target_sentence_idx)]

def language_pair_collate_func(batch, max_sentence_length):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length, max_sentence_length
    """
    source_list = []
    target_list = []
    source_len_list = []
    target_len_list = []
    
    # Add 1 for EOS
    for datum in batch:
        source_len_list.append(datum[2] + 1)
        target_len_list.append(datum[3] + 1)
        
    # Append EOS_IDX to the end of the sentences and pad
    for datum in batch:
        padded_source_vec = np.pad(np.array(datum[0] + [EOS_IDX]), 
                                pad_width=((0,max_sentence_length + 1 - (datum[2] + 1))), 
                                mode="constant", constant_values=0)
        padded_target_vec = np.pad(np.array(datum[1] + [EOS_IDX]),
                                  pad_width=((0, max_sentence_length + 1 - (datum[3] + 1))),
                                  mode="constant", constant_values=0)
        
        source_list.append(padded_source_vec)
        target_list.append(padded_target_vec)
    return [torch.from_numpy(np.array(source_list)), torch.from_numpy(np.array(target_list)), torch.LongTensor(source_len_list), torch.LongTensor(target_len_list)]




# Create language pair data loaders
BATCH_SIZE = 32

# Vietnamese to English datasets
vi_to_en_train_dataset = LanguagePairs(vi_to_en_vi_train_indices, vi_to_en_en_train_indices, MAX_SENTENCE_LENGTH_VI_EN)
vi_to_en_val_dataset = LanguagePairs(vi_to_en_vi_val_indices, vi_to_en_en_val_indices, MAX_SENTENCE_LENGTH_VI_EN)
vi_to_en_test_dataset = LanguagePairs(vi_to_en_vi_test_indices, vi_to_en_en_test_indices, MAX_SENTENCE_LENGTH_VI_EN)

# Chinese to English datasets
zh_to_en_train_dataset = LanguagePairs(zh_to_en_zh_train_indices, zh_to_en_en_train_indices, MAX_SENTENCE_LENGTH_ZH_EN)
zh_to_en_val_dataset = LanguagePairs(zh_to_en_zh_val_indices, zh_to_en_en_val_indices, MAX_SENTENCE_LENGTH_ZH_EN)
zh_to_en_test_dataset = LanguagePairs(zh_to_en_zh_test_indices, zh_to_en_en_test_indices, MAX_SENTENCE_LENGTH_ZH_EN)

# Vietnamese to English dataloaders
vi_to_en_train_loader = torch.utils.data.DataLoader(dataset=vi_to_en_train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_VI_EN),
                                           shuffle=True)

vi_to_en_val_loader = torch.utils.data.DataLoader(dataset=vi_to_en_val_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_VI_EN),
                                                 shuffle=True)

vi_to_en_test_loader = torch.utils.data.DataLoader(dataset=vi_to_en_test_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_VI_EN),
                                                 shuffle=False)

# Chinese to English dataloaders
zh_to_en_train_loader = torch.utils.data.DataLoader(dataset=zh_to_en_train_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_ZH_EN),
                                                 shuffle=True)

zh_to_en_val_loader = torch.utils.data.DataLoader(dataset=zh_to_en_val_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_ZH_EN),
                                                 shuffle=True)

zh_to_en_test_loader = torch.utils.data.DataLoader(dataset=zh_to_en_test_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 collate_fn=lambda x: language_pair_collate_func(x, MAX_SENTENCE_LENGTH_ZH_EN),
                                                 shuffle=False)


# for source, target, source_len, target_len in zh_to_en_train_loader:
#     print(source.shape)
#     break


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Create embedding matrix
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Create gated-recurrent unit
        if bidirectional:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)
                
        
    def sort_batch(self, sents, lengths):
        """
        @param: sents - source sentences to translate
        @param: lengths - lengths of sources sentences
        
        returns the sentences in descending order, the lengths in descending order,
        and the order of the indices corresponding to descending order"""
        
        lengths = lengths
        sents = sents
        
        ind_dec_order = np.argsort(lengths.numpy())[::-1]
        lens_desc = lengths.numpy()[ind_dec_order]
        lens_desc = torch.from_numpy(lens_desc)
        sents_desc = sents.numpy()[ind_dec_order]
        sents_desc = torch.from_numpy(sents_desc)
        
        sents_desc = sents_desc.to(device)
        lens_desc = lens_desc.to(device)
        
        return sents_desc, lens_desc, ind_dec_order
    
    # Sort outputs back into original order of the batch
    def unsort_outputs(self, outputs, ordering):
        """
        @param: outputs - outputs from RNN encoding
        @param: ordering - ind_dec_order, the ordering of indices corresponding to sorting
        by decreasing length"""
        original_order = np.argsort(ordering)
        outputs_original = outputs[original_order]
        
        return outputs_original
    
    # Sort hidden states back into original order of the batch
    def unsort_hidden(self, last_hidden, ordering):
        #original_size = last_hidden.size()
        original_order = np.argsort(ordering)

        
        
        last_hidden_original = last_hidden[:, original_order, :]
        return last_hidden_original
    
    # Initialize hidden state
    def init_hidden(self, batch_size):
        if self.bidirectional:
            hidden = torch.randn(2 * self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            hidden = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            
        return hidden
    
    # Forward propagate through the RNN
    def forward(self, source_sentences, source_lengths):
        # Initialize hidden state
        self.hidden = self.init_hidden(source_sentences.shape[0])
        
        # Sort source sentences
        source_sentences_desc, lens_desc, ind_dec_order = self.sort_batch(source_sentences, source_lengths)
        
        # Get embedding
        res = self.embedding(source_sentences_desc)
        
        res = torch.nn.utils.rnn.pack_padded_sequence(res, lens_desc, batch_first=True)
        packed_output, last_hidden = self.gru(res, self.hidden)
        
        # Unpack output
        unpacked_outputs, lens = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Concatenate last_hidden if bidirectional
        if self.bidirectional:
            forward = last_hidden[0]
            backward = last_hidden[1]

            last_hidden = torch.cat((forward, backward), dim=1).view(1, source_sentences.shape[0], -1)
            
        
        # Sort unpacked_output and last_hidden states into original order
        outputs = self.unsort_outputs(unpacked_outputs, ind_dec_order)
        last_hidden = self.unsort_hidden(last_hidden, ind_dec_order)
        
        # Return outputs and last_hidden states corresponding to their original order
        return outputs, last_hidden


# From lab 8 - RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, bidirectional_inputs=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional_inputs = bidirectional_inputs

        self.embedding = nn.Embedding(output_size, hidden_size)
        
        if self.bidirectional_inputs:
            self.gru = nn.GRU(hidden_size, 2 * hidden_size, batch_first=True)
            self.out = nn.Linear(2 * hidden_size, output_size)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.out = nn.Linear(hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=2)

    # Forward propagate. Does not use encoder_outputs.
    # Returns 0 to return same number of items as decoder with
    # attention.
    def forward(self, input, hidden, encoder_outputs):
        # Input is a single word
        #print(type(input))
        output = self.embedding(input)
        
        # This should be BATCH_SIZE x 1 x hidden_size
        # print("DECODER EMBEDDING", output.shape)
        
        output = F.relu(output)

        # print(output.size())
        # print(hidden.size())
        output, hidden = self.gru(output, hidden)
        # print("OUTPUT FROM GRU", output.shape)
        output = self.out(output)
        # print("OUTPUT FROM LINEAR", output.shape)
        output = self.softmax(output)
        # print("OUTPUT FROM SOFTMAX", output.shape)
        return output, hidden, 0


# From lab 8 - RNN Decoder with Attention
# Always takes in bidirectional inputs
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, attention_mechanism='dot', dropout_p=0.1, rnn='gru'):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.attention_mechanism = attention_mechanism
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        # For general attention mechanism
        if attention_mechanism == 'general':
            self.attn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        elif attention_mechanism == 'concat':
            self.attn = nn.Linear(4 * hidden_size, hidden_size)
            self.v = nn.Parameter(torch.randn(hidden_size, 1))
        
        # Should be the same regardless of attention mechanism
        if rnn == 'gru':
            self.rnn = nn.GRU(hidden_size * 3, hidden_size * 2, n_layers, dropout=dropout_p, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size * 3, hidden_size * 2, n_layers, dropout=dropout_p, batch_first=True)
                
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # print("ENCODER OUTPUT SHAPES", encoder_outputs.shape)
        # print("LAST HIDDEN SHAPES", last_hidden.shape)
        # encoder_outputs should be BATCH_SIZE x Seq Len x (2 x hidden_size)
        # last_hidden should be 1 x BATCH_SIZE x (2 x hidden_size)
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input) # This should be BATCH_SIZE x 1 x hidden_size
        
        # Pass through dropout layer
        word_embedded = self.dropout(word_embedded)
        
        ##############################
        # TODO: Implement Attention  #
        ##############################

        # Get batch size
        batch_size = last_hidden.shape[1]
        seq_length = encoder_outputs.shape[1]
        
        if self.attention_mechanism == 'dot':
            reshape_hidden = last_hidden.view(batch_size, 1, -1)
            reshape_encoder_outputs = encoder_outputs.transpose(2, 1)
            scores = torch.matmul(reshape_hidden, reshape_encoder_outputs)
            # Scores should be BATCH_SIZE x 1 x Seq Len
            attn_weights = F.softmax(scores, dim=2)
        elif self.attention_mechanism == 'general':
            # Pass encoder_outputs through linear network
            # Dot product with h_t to compute scores
            reshape_hidden = last_hidden.view(batch_size, 1, -1)
            encoder_outputs = self.attn(encoder_outputs)
            reshape_encoder_outputs = encoder_outputs.transpose(2, 1)
            scores = torch.matmul(reshape_hidden, reshape_encoder_outputs)
            attn_weights = F.softmax(scores, dim=2)

        else:
            # Concatenation attention
            
            # Reshape last_hidden
            reshape_hidden = last_hidden.view(batch_size, 1, -1)
            # Repeat last_hidden for concatenation
            hidden_repeat = reshape_hidden.repeat(1, seq_length, 1)
            # Concatenation (BATCH SIZE x Seq Len x (2 x hidden_size))
            concat = torch.cat((encoder_outputs, hidden_repeat), dim=2)
            # Pass concatenation through linear network
            concat = self.attn(concat)
            # Output (BATCH SIZE x Seq Len x hidden_size)
            
            scores = torch.matmul(concat, self.v)
            scores = scores.view(batch_size, 1, -1)
            scores = nn.tanh(scores)
            attn_weights = F.softmax(scores, dim=2)
        # Create context vector
        context = torch.matmul(attn_weights, encoder_outputs)

        # Reshape context vector to feed into GRU
        context = context.view(1, batch_size, -1)
        # Context should be shape (1 x BATCH_SIZE x (2 x hidden_size))
        
        rnn_input = torch.cat((context, word_embedded.view(1, batch_size, -1)), dim=2)
        # rnn_input should be 1 x BATCH_SIZE x (3 x hidden_size)
        # Reshape rnn_input for GRU
        rnn_input = rnn_input.view(batch_size, 1, -1)
        
        
        output, hidden = self.rnn(rnn_input, last_hidden)
        
        output = self.out(output)
        
        output = self.softmax(output)
            

        return output, hidden, attn_weights

# Evaluate models and return loss and BLEU score
def evaluate(data_loader, encoder, decoder, criterion, k):
    """
    @param: train_loader - training data loader
    @param: val_loader - validation data loader
    @param: encoder - encoder to use
    @param: decoder - decoder to use
    @param: criterion - loss function
    @param: k - beam size for beam search
    """
    
    
    # Compute loss
    encoder.eval()
    decoder.eval()
    cumulative_loss = 0
    
    # torch.no_grad
    
    for i, (source_sents, target_sents, source_lengths, target_lengths) in enumerate(data_loader):

        BATCH_SIZE = source_sents.size(0)

        encoder_outputs, last_hidden_states = encoder(source_sents, source_lengths)
        # Create <SOS> tensor to feed into decoder
        decoder_input = SOS_IDX * torch.ones(BATCH_SIZE, 1, dtype=torch.long)
            
        # First decoder hidden
        decoder_hidden = last_hidden_states
        
        decoder_input = decoder_input.to(device)
        decoder_hidden = decoder_hidden.to(device)

        mask = target_sents > 0
        
        batch_loss = 0
        
        for j in range(target_sents.shape[1]):
            next_words = target_sents[:, j].squeeze().to(device)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
                    
            # Change to sum and divide by total number of non-padded idx for loss
            loss = criterion(decoder_output.squeeze(), next_words)
            batch_loss += loss.item()
            topv, topi = decoder_output.topk(1, dim=2)
            next_words = topi.view(-1, 1).detach()
            decoder_input = next_words
        
        cumulative_loss +=  batch_loss / mask.sum().item()
    
    return cumulative_loss/len(data_loader)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





def initialize_beam_search(k, decoder, decoder_input, decoder_hidden, encoder_outputs, id2token, max_length):
    """
    Function to initialize the beam search
    @ param: k - beam size
    @param: decoder - the decoder to use
    @param: decoder_input - initial decoder input [SOS] token
    @param: decoder_hidden - initial decoder hidden state. This is the last hidden state returned by the encoder.
    @param: encoder_outputs - outputs from the encoder
    @param: max_length - the maximum number of words that the decoder can return
    """
    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
    decoder_output = decoder_output.squeeze()
    topv, topi = decoder_output.topk(k)
    topi = topi.squeeze()
    candidates = []

    for i in range(len(topi)):
        word = id2token[topi[i].item()]
        score = topv[i].item()
        next_input = topi[i].squeeze().detach()
        candidates.append(([word], score, decoder_hidden, next_input, decoder_attention))
    return beam_search(k, candidates, decoder, encoder_outputs, id2token, max_length)

def beam_search(k, candidates, decoder, encoder_outputs, id2token, max_length):
    """
    Function that continues beam search.
    @param: k - beam size
    @param: candidates - a list of tuples containing (list_of_words, score, last_hidden, next_input, attention_matrix)
    @param: max_length - the maximum number of words that the decoder can return
    @param: encoder_outputs - the list of outputs from the encoder
    """
    translations = list(map(lambda cand: cand[:2], candidates))

    # We know we have k candidates
    # Check to see if all of the next_inputs are EOS
    next_inputs = list(map(lambda cand: cand[3].item(), candidates))
    next_inputs = list(set(next_inputs))
    if len(next_inputs) == 1 and EOS_IDX in next_inputs:
        # Get best based on score
        sorted_by_score = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        return sorted_by_score[0][0], sorted_by_score[0][-1].squeeze()

    
    # Get maximum decoded sentence
    # Sort candidates
    sorted_candidates = sorted(candidates, key=lambda candidate: len(candidate[0]), reverse=True)
    candidate_max_length = len(sorted_candidates[0][0])
    if candidate_max_length == max_length:
        # Return best translation based on score
        sorted_by_score = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        return sorted_by_score[0][0], sorted_by_score[0][-1].squeeze()
        
    else:
        # Continue
        new_candidates = []
        for candidate in candidates:
            if candidate[3] == EOS_IDX:
                new_candidates.append(candidate)
            else:
                
                decoder_input = candidate[3] * torch.ones(1, 1, dtype=torch.long)
                
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, candidate[2], encoder_outputs)
                decoder_output = decoder_output.squeeze()
                topv, topi = decoder_output.topk(k)
                topi = topi.squeeze()
                for i in range(len(topi)):
                    word = id2token[topi[i].item()]
                    cur_translation = candidate[0].copy()
                    cur_translation.append(word)
                    score = topv[i].item()
                    new_score = score + candidate[1]
                    next_input = topi[i].squeeze().detach()
                    #print(candidate[4])
                    cur_attention = candidate[4].clone()
                    cur_attention = torch.cat((cur_attention, decoder_attention), dim=0)
                    new_candidates.append((cur_translation, new_score, decoder_hidden, next_input, cur_attention))
        # Now have a list of candidates
        sorted_candidates = sorted(new_candidates, key=lambda candidate: candidate[1], reverse=True)
        new_candidates = sorted_candidates[:5]
        
        return beam_search(k, new_candidates, decoder, encoder_outputs, id2token, max_length)

def greedy_search(decoder, decoder_input, decoder_hidden, encoder_outputs, max_length, decoder_attentions, decoded_words, id2token):
    """
    Function that decodes using a greedy search.
    @param: max_length - the maximum number of words that the decoder can return
    @param: decoder_input - the inital input for the decoding process [SOS] token
    @param: decoder_hidden - the initial hidden state for the decoder, given by last hidden state of the encoder
    @param: encoder_outputs - the list of outputs from the encoder
    @param: decoder_attentions - an empty matrix to store the attentions
    @param: decoded_words - an empty list to store the decoded words
    """
    with torch.no_grad():
        for di in range(max_length):
            # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # hint: print out decoder_output and decoder_attention
            # TODO: add your code here to populate decoded_words and decoder_attentions
            # TODO: do this in 2 ways discussed in class: greedy & beam_search
            # Add decoder attentions to matrix
            decoder_attentions[di, :] = decoder_attention

            # Greedy search to get word
            topv, topi = decoder_output.topk(1, dim=2)
            word = id2token[topi.item()]
            decoded_words.append(word)
            if topi == EOS_IDX:
                break

            decoder_input = topi.view(-1, 1).detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_BLEU(encoder, decoder, sentence, id2token, max_length=MAX_LENGTH):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """    
    # process input sentence
    encoder.eval()
    decoder.eval()
    criterion = nn.NLLLoss(ignore_index=0, reduction='sum')


    input_tensor = sentence
    input_length = torch.tensor([input_tensor.size()[1]])


    encoder_outputs, last_hidden_states = encoder(input_tensor, input_length)

    # Create <SOS> tensor to feed into decoder
    decoder_input = SOS_IDX * torch.ones(1, 1, dtype=torch.long)
        
    # First decoder hidden
    decoder_hidden = last_hidden_states
    
    decoder_input = decoder_input.to(device)
    decoder_hidden = decoder_hidden.to(device)

    #mask = target_sents > 0
    
    batch_loss = 0
    decoded_words = []
    
    for j in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        # Change to sum and divide by total number of non-padded idx for loss
        topv, topi = decoder_output.topk(1, dim=2)
        #print(topi.shape)
        word = id2token[topi.item()]
        decoded_words.append(word)
        if topi == EOS_IDX:
            break
        next_words = topi.view(-1, 1).detach()
        decoder_input = next_words
    
    #cumulative_loss +=  batch_loss / mask.sum().item()

    return decoded_words

    # with torch.no_grad():
    #     input_tensor = sentence
    #     input_length = torch.tensor([input_tensor.size()[1]])
    #     # encode the source lanugage
    #     encoder_hidden = encoder.init_hidden(1)

    #     #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    #     #for ei in range(input_length):
    #     encoder_outputs, encoder_hidden = encoder(input_tensor,
    #                                              input_length)
    #         #encoder_outputs[ei] += encoder_output[0, 0]

    #     decoder_input = torch.tensor([[SOS_IDX]], device=device)  # SOS
    #     # decode the context vector
    #     decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
    #     # output of this function
    #     decoded_words = []
    #     decoder_attentions = torch.zeros(max_length, max_length).to(device)

    #     decoded_words, decoder_attentions = greedy_search(decoder, decoder_input, decoder_hidden, encoder_outputs, max_length, decoder_attentions, decoded_words, id2token)
    #     #decoded_words, decoder_attentions = initialize_beam_search(10, decoder, decoder_input, decoder_hidden, encoder_outputs, id2token, max_length)
    #     return decoded_words, decoder_attentions



import sacrebleu

# cands = ["This is sentence one", "This is sentence two", "This is sentence three"]
# targets = ["This is sentence one", "This is sentence two", "This is sentence three"]
# bleu = sacrebleu.corpus_bleu(cands, [targets])

# print(bleu)



def compute_bleu(encoder, decoder, test_loader, source_id2token, id2token, max_length = MAX_LENGTH):

    target_stream = []
    decoded_stream = []
    print(len(test_loader))
    for idx, (source_sents, target_sents, source_lengths, target_lengths) in enumerate(test_loader):
        print(idx)
        for i in range(len(source_sents)):
            sentence = source_sents[i]
            sentence = sentence.view(1, -1)
            decoded_words = evaluate_BLEU(encoder, decoder, sentence, id2token, max_length)
            target_words = []
            for j in range(len(target_sents[i])):
                if target_sents[i][j] == EOS_IDX: 
                    break
                target_words.append(id2token[target_sents[i][j]])
            # print('>:  ' + ' '.join(target_words))
            # print('<:  ' + ' '.join(decoded_words[: -1]))
            # print()
            target_stream.append(' '.join(target_words))
            decoded_stream.append(' '.join(decoded_words[: -1]))

    return sacrebleu.corpus_bleu(target_stream, [decoded_stream])
        
#print('BLEU score: ' + str(compute_bleu("vi_en", vi_to_en_test_loader, vi_to_en_vi_id2token, vi_to_en_en_id2token)))


epoch = 2

lang_pair = 'vi_en'

encoder = EncoderRNN(len(vi_to_en_vi_id2token), 300, 1, bidirectional=True).to(device)
decoder = DecoderRNN(300, len(vi_to_en_en_id2token), bidirectional_inputs = True).to(device)

#criterion = nn.NLLLoss(ignore_index=0, reduction='sum')

encoder.load_state_dict(torch.load("simple_encoder_batched_" + lang_pair + '_epoch_' + str(epoch)+ ".pth", map_location='cpu'))
decoder.load_state_dict(torch.load("simple_decoder_batched_" + lang_pair+ '_epoch_' + str(epoch) + ".pth", map_location='cpu'))



print(str(compute_bleu(encoder, decoder, vi_to_en_val_loader, vi_to_en_vi_id2token, vi_to_en_en_id2token)))
