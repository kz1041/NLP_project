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
import math

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
            
# Load training set for Vietnamese-English
vi_en_train = create_dataframe('./iwslt-vi-en/train.tok.vi', './iwslt-vi-en/train.tok.en', 'vi', 'en')

# Load training set for Chinese-English
zh_en_train = create_dataframe('./iwslt-zh-en/train.tok.zh', './iwslt-zh-en/train.tok.en', 'zh', 'en')

# Load validation set for Vietnamese-English
vi_en_val = create_dataframe('./iwslt-vi-en/dev.tok.vi', './iwslt-vi-en/dev.tok.en', 'vi', 'en')

# Load validation set for Chinese-English
zh_en_val = create_dataframe('./iwslt-zh-en/dev.tok.zh', './iwslt-zh-en/dev.tok.en', 'zh', 'en')

# Load test set for Vietnamese-English
vi_en_test = create_dataframe('./iwslt-vi-en/test.tok.vi', './iwslt-vi-en/test.tok.en', 'vi', 'en')

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
MAX_VOCAB_SIZE = 10000
max_vocab_size_zh_en_en = 6000
max_vocab_size_zh_en_zh = 12000
max_vocab_size_vi_en_en = 10000
max_vocab_size_vi_en_vi = 10000


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
zh_to_en_en_token2id, zh_to_en_en_id2token = build_vocabulary(zh_en_train_en_all_tokens, max_vocab_size_zh_en_en)

# Chinese-to-English Chinese id2token and token2id
zh_to_en_zh_token2id, zh_to_en_zh_id2token = build_vocabulary(zh_en_train_zh_all_tokens, max_vocab_size_zh_en_zh)

# Vietnamese-to-English English id2token and token2id
vi_to_en_en_token2id, vi_to_en_en_id2token = build_vocabulary(vi_en_train_en_all_tokens, max_vocab_size_vi_en_en)

# Vietnamese-to-English Vietnamese id2token and token2id
vi_to_en_vi_token2id, vi_to_en_vi_id2token = build_vocabulary(vi_en_train_vi_all_tokens, max_vocab_size_vi_en_vi)





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



# Batch implementation of models

# From lab 8 - Forward RNN Encoder
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
    def __init__(self, hidden_size, output_size, bidirectional_inputs=False, dropout_p = 0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional_inputs = bidirectional_inputs

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
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
    def forward(self, input, hidden, encoder_outputs, last_hidden, mask):
        # Input is a single word
        #print(type(input))
        if hidden is None:
            hidden = last_hidden
        output = self.embedding(input)
        
        # This should be BATCH_SIZE x 1 x hidden_size
        # print("DECODER EMBEDDING", output.shape)
        output = self.dropout(output)
        #output = F.relu(output)

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
        self.attn = nn.Linear(2*hidden_size, 2*hidden_size)
        # Should be the same regardless of attention mechanism
        if rnn == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size*2 , n_layers, dropout=dropout_p, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size * 3, hidden_size * 2, n_layers, dropout=dropout_p, batch_first=True)
                
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        self.Whc = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, word_input, hidden, encoder_outputs, last_hidden, mask):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # print("ENCODER OUTPUT SHAPES", encoder_outputs.shape)
        # print("LAST HIDDEN SHAPES", last_hidden.shape)
        # encoder_outputs should be BATCH_SIZE x Seq Len x (2 x hidden_size)
        # last_hidden should be 1 x BATCH_SIZE x (2 x hidden_size)
        # Get the embedding of the current input word (last output word)
        if hidden is None:
            hidden = last_hidden

        batch_size = word_input.size(0)

        word_embedded = self.embedding(word_input) # This should be BATCH_SIZE x 1 x hidden_size
        
        # Pass through dropout layer
        word_embedded = self.dropout(word_embedded)
        #word_embedded = F.relu(word_embedded)

        rnn_output, hidden = self.rnn(word_embedded, hidden)
        # print(rnn_output.shape)
        # print(hidden.shape)
        ##############################
        # TODO: Implement Attention  #
        ##############################
        reshape_hidden = rnn_output.view(batch_size, 1, -1)
        reshape_encoder_outputs = encoder_outputs.transpose(2, 1)
        # print(hidden.shape)
        # print(reshape_encoder_outputs.shape)
        scores = torch.matmul(self.attn(reshape_hidden), reshape_encoder_outputs)

        mask = mask[:, :scores.size(2)].view(batch_size, 1, -1).to(device)
        scores.data.masked_fill_(mask == 0, -float('inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        #print(scores.shape)

        context = torch.matmul(attn_weights, encoder_outputs)
        # Reshape context vector to feed into GRU
        context = context.view(1, batch_size, -1)
        # Context should be shape (1 x BATCH_SIZE x (2 x hidden_size))
        # print(context.shape)
        # print(hidden.shape)
        rnn_output = rnn_output.view(1, batch_size, -1)

        output = F.tanh(self.Whc(torch.cat([rnn_output, context], dim=-1)))

        output = self.out(output)
        output = self.softmax(output)
            
        return output.view(1, batch_size, -1), hidden, attn_weights





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
        decoder_hidden = None
        
        decoder_input = decoder_input.to(device)

        mask = target_sents > 0
        
        batch_loss = 0
        
        for j in range(target_sents.shape[1]):
            next_words = target_sents[:, j].squeeze().to(device)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_hidden_states, source_sents>0)
                    
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
    decoder_hidden = None

    decoder_input = decoder_input.to(device)

    #mask = target_sents > 0
    
    batch_loss = 0
    decoded_words = []
    
    for j in range(max_length):
        #print(sentence>0)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, last_hidden_states, sentence>0)
        # Change to sum and divide by total number of non-padded idx for loss
        topv, topi = decoder_output.topk(1, dim=2)
        #print(topi.shape)
        word = id2token[topi.item()]
        decoded_words.append(word)
        if topi == EOS_IDX:
            break
        next_words = topi.view(-1, 1).detach()
        decoder_input = next_words
        # if j >0: continue
        print('j = ' + str(j))
        print(decoder_attention)
    #cumulative_loss +=  batch_loss / mask.sum().item()

    return decoded_words



import sacrebleu

# cands = ["This is sentence one", "This is sentence two", "This is sentence three"]
# targets = ["This is sentence one", "This is sentence two", "This is sentence three"]
# bleu = sacrebleu.corpus_bleu(cands, [targets])

# print(bleu)



def compute_bleu(encoder, decoder, test_loader, source_id2token, id2token, max_length = MAX_LENGTH):

    target_stream = []
    decoded_stream = []
    for idx, (source_sents, target_sents, source_lengths, target_lengths) in enumerate(test_loader):
        #print(idx)
        for i in range(len(source_sents)):
            sentence = source_sents[i][source_sents[i]>0]
            sentence = sentence.view(1, -1)
            decoded_words = evaluate_BLEU(encoder, decoder, sentence, id2token, max_length)
            target_words = []
            source_words = []
            for j in range(len(source_sents[i])):
                if source_sents[i][j] == EOS_IDX: 
                    break
                source_words.append(source_id2token[source_sents[i][j]])
            for j in range(len(target_sents[i])):
                if target_sents[i][j] == EOS_IDX: 
                    break
                target_words.append(id2token[target_sents[i][j]])
            # print('> ' + ' '.join(source_words))
            # print('> ' + ' '.join(target_words))
            # print('< ' + ' '.join(decoded_words[: -1]))
            target_stream.append(' '.join(target_words))
            decoded_stream.append(' '.join(decoded_words[: -1]))

    return sacrebleu.corpus_bleu(target_stream, [decoded_stream])
        
#print('BLEU score: ' + str(compute_bleu("vi_en", vi_to_en_test_loader, vi_to_en_vi_id2token, vi_to_en_en_id2token)))

# lang_pair = 'vi_en'
# encoder = EncoderRNN(len(source_id2token), 300, 1, bidirectional=True).to(device)
# decoder = DecoderRNN(300, len(id2token), bidirectional_inputs = True).to(device)

# criterion = nn.NLLLoss(ignore_index=0, reduction='sum')


# encoder.load_state_dict(torch.load("attention_encoder_batched_" + lang_pair + ".pth"))
# decoder.load_state_dict(torch.load("attention_decoder_batched_" + lang_pair + ".pth"))



def trainEpoch(train_loader, val_loader, source_id2token, target_id2token, lang_pair, num_epochs = 5, learning_rate = 0.0002, decoder_learning_ratio = 5.0, decay = 0.9, decay_epoch = 1, bleu_epoch = 1):
    print('Total Num Epochs: ' + str(num_epochs))

    rnn_encoder = EncoderRNN(len(source_id2token), 300, 1, bidirectional=True).to(device)
    #rnn_encoder = EncoderRNN2(len(source_id2token), 300,  bidirectional=False).to(device)

    #rnn_decoder = DecoderRNN(300, len(target_id2token), bidirectional_inputs = True).to(device)
    #rnn_decoder = DecoderRNN(300, len(target_id2token), bidirectional_inputs=True).to(device)
    rnn_decoder = AttnDecoderRNN(300, len(target_id2token), attention_mechanism = 'dot').to(device)


    learning_rate = learning_rate
    encoder_optimizer = optim.Adam(rnn_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(rnn_decoder.parameters(), lr=learning_rate*decoder_learning_ratio)
    criterion = nn.NLLLoss(ignore_index=0, reduction='sum')


    #batch = next(iter(zh_to_en_train_loader))

    
    train_loader = train_loader
    val_loader = val_loader

    # train_loader = [batch]
    # val_loader = [batch]

    encoder = rnn_encoder
    decoder = rnn_decoder


    # encoder.load_state_dict(torch.load("simple_encoder_batched_" + lang_pair + ".pth"))
    # decoder.load_state_dict(torch.load("simple_encoder_batched_" + lang_pair + ".pth"))



    num_epochs = num_epochs
    teacher_forcing_ratio = 1.0



    for epoch in range(num_epochs):
        if epoch % decay_epoch == 0:
            learning_rate = learning_rate * decay
            for param_group in encoder_optimizer.param_groups:
                param_group['lr'] = learning_rate
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = learning_rate * decoder_learning_ratio


        for i, (source_sents, target_sents, source_lengths, target_lengths) in enumerate(train_loader):
            #print(i)
            encoder.train()
            decoder.train()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Get encoder_outputs (B x Seq Len X Hidden Size) and last_hidden_states (1 X B X Hidden Size)
            encoder_outputs, last_hidden_states = encoder(source_sents, source_lengths)

            # Stores loss
            loss = 0
            BATCH_SIZE = source_sents.size(0)
            # Create <SOS> tensor to feed into decoder
            decoder_input = SOS_IDX * torch.ones(BATCH_SIZE, 1, dtype=torch.long)


            # First decoder hidden

            # Use teacher forcing for this batch
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            
            decoder_input = decoder_input.to(device)
            decoder_hidden = None

            all_decoder_outputs = []
            # Does it matter if I stop training when I get EOS?
            if use_teacher_forcing:
                # Iterate over target_sents
                for j in range(target_sents.shape[1]):
                    next_words = target_sents[:, j].squeeze().to(device)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs, last_hidden_states, source_sents>0)
                    loss += criterion(decoder_output.squeeze(), next_words)
                    decoder_input = next_words.view(-1, 1)
                    all_decoder_outputs.append(next_words)
            else:
                for j in range(target_sents.shape[1]):
                    next_words = target_sents[:, j].squeeze().to(device)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    # print('j: ' + str(j))
                    # if j == 0:
                    #     print(decoder_attention[0, :, :])
                    # if j == 10:
                    #     print(decoder_attention[0, :, :])
                    loss += criterion(decoder_output.squeeze(), next_words)
                    topv, topi = decoder_output.topk(1)
                    next_words = topi.view(-1, 1).detach()
                    decoder_input = next_words
                    all_decoder_outputs.append(next_words)
            
            mask = target_sents > 0
            loss = loss / mask.sum().item()

            train_loss =loss

            
            # Evaluate 
            if i % 100 == 0:
                #train_loss = evaluate(train_loader, encoder, decoder, criterion, 1)
                val_loss = evaluate(val_loader, encoder, decoder, criterion, 1)

                print('Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}, Val Loss: {}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), train_loss, val_loss))

            # Compute gradients
            loss.backward()

            # Update weights in encoder and decoder
            clip = 50.0
            # Clip gradient norms
            ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
            dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

            encoder_optimizer.step()
            decoder_optimizer.step()

        torch.save(encoder.state_dict(), "attn_encoder_batched_" + lang_pair + '_epoch_' + str(epoch)+ ".pth")
        torch.save(decoder.state_dict(), "attn_decoder_batched_" + lang_pair+ '_epoch_' + str(epoch) + ".pth")
        if epoch % bleu_epoch == 0: 
            print(compute_bleu(encoder, decoder, val_loader, source_id2token, target_id2token, max_length = MAX_LENGTH))

    # torch.save(encoder.state_dict(), "attn_encoder_batched_" + lang_pair  + ".pth")
    # torch.save(decoder.state_dict(), "attn_decoder_batched_" + lang_pair + ".pth")


some_batches = []
some_batches.append(next(iter(vi_to_en_train_loader)))
# some_batches.append(next(iter(vi_to_en_train_loader)))
# some_batches.append(next(iter(vi_to_en_train_loader)))
# some_batches.append(next(iter(vi_to_en_train_loader)))
# some_batches.append(next(iter(vi_to_en_train_loader)))

trainEpoch(vi_to_en_train_loader, vi_to_en_val_loader, vi_to_en_vi_id2token, vi_to_en_en_id2token, 'vi_en', num_epochs = 16)
#






