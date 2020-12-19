# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import numpy as np # for np.multiply

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        '''Additions'''
        self.data = data
        '''End Additions'''
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel-1)/2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
        We do the neuron ablation in this function, by editing these lines of code:

        # change feature_order to be your list of neurons to ablate in order of importance
        feature_order = \
            [44, 12, 32, 0, 28, 34, 14, 40, 16, 43, 42, 35, 41, 36, 7, 47, 49, 5, 1, 31, 24, 8, 6, 23, 22, 37, 10, 39,
             27, 26, 25, 13, 48, 46, 21, 18, 38, 9, 17, 33, 20, 4, 19, 29, 11, 2, 15, 3, 45, 30]
        mask_np = np.ones(feature_out.shape)
        mask_np[:, :, feature_order[0:0]] = 0

        To ablate the 1st 25 neurons, mask_np[:, :, feature_order[0:25]] = 0

            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
    
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            # np_lstm_out = lstm_out.cpu().detach().numpy()
            np_lstm_out_trans = lstm_out.transpose(1,0).cpu().detach().numpy()
            ## lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)


        ''' Ablation of neurons '''
        ## this is the feature_order of a tag like B-ORG, you can change it to what you get.
        # Example for b-org g_100:
        '''
        feature_order = \
            [7, 41, 49, 39, 2, 17, 33, 10, 31, 34, 29, 23, 15, 27, 24, 47, 44, 42, 30, 22, 21, 11, 0, 12, 4, 8, 20, 9,
             5, 48, 19, 45, 32, 43, 3, 35, 36, 25, 13, 16, 37, 38, 28, 46, 14, 18, 6, 26, 1, 40]
        '''
        feature_order = self.data.ablate_list[self.data.ablate_tag]

        # if we don't have enough neurons in the list to ablate the number specified in data.ablate_num
        # then change the ablate num
        if len(self.data.ablate_list[self.data.ablate_tag]) < self.data.ablate_num:
            print("\nWARNING: ABLATION LIST FOR TAG: {} has length {}, ablate_num={}. Changing ablate num to match len".format(
                self.data.ablate_tag, len(self.data.ablate_list[self.data.ablate_tag]),
                self.data.ablate_num
            ))
            self.data.ablate_num = len(self.data.ablate_list[self.data.ablate_tag])


        mask_np = np.ones(feature_out.shape)
        if self.data.ablate_num > 0:
            mask_np[:, :, feature_order[0:self.data.current_ablate_ind[self.data.ablate_tag]]] = 0   ## For 1st 25, mask_np[:, :, feature_order[0:25]] = 0
        else:
            mask_np[:, :, feature_order[0:0]] = 0   ## For 1st 25, mask_np[:, :, feature_order[0:25]] = 0

        mask_tensor = torch.from_numpy(mask_np)
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
        device = torch.device("cuda")
        mask_tensor = mask_tensor.to(device)
        feature_out = feature_out.mul(mask_tensor)
        outputs = self.hidden2tag(feature_out)

        ''' SAVE THE WEIGHTS '''
        np_weights = self.hidden2tag.weight.cpu().detach().numpy()
        if self.data.weights_saved == False:
            self.data.weights_saved == True
            self.data.weights = np_weights
            np.save('weights.npy', np_weights)
        self.data.iteration += 1  # counts number of batches

        ''' SAVE CONTRIBUTIONS FOR THIS BATCH '''
        self.data.batch_contributions = [[np.multiply(
            np_weights, np_lstm_out_trans[i][j])
                for j in range(len(np_lstm_out_trans[i]))] for i in range(len(np_lstm_out_trans))]

        flat_lstm_out = lstm_out.view(word_inputs.size(0) * max(word_seq_lengths), -1)

        return outputs


    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        batch_size = word_inputs.size(0)
        if self.word_feature_extractor == "CNN":
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = F.max_pool1d(cnn_feature, cnn_feature.size(2)).view(batch_size, -1)
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            ## lstm_out (seq_len, seq_len, hidden_size)
            ## feature_out (batch_size, hidden_size)
            feature_out = hidden[0].transpose(1,0).contiguous().view(batch_size,-1)
            
        feature_list = [feature_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        return outputs
