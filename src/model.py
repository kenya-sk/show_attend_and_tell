#! /usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_variable, tensor2numpy


class EncoderResNet(nn.Module):
    def __init__(self, encoded_img_size=14, model_path=None):
        super(EncoderResNet, self).__init__()

        # feature extraction model (ResNet152)
        resnet = models.resnet152(pretrained=True)
        self._resnet_extractor = nn.Sequential(*(list(resnet.children())[:-2]))

        # Resize image (encoded_img_size)
        self.encoded_img_size = encoded_img_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size, encoded_img_size))

    def forward(self, x):
        feature = self._resnet_extractor(x)
        feature = self.adaptive_pool(feature)
        return feature
    
    
class Decoder(nn.Module):
    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()
        
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        #self.init_h = nn.Linear(vis_dim, hidden_dim)
        #self.init_c = nn.Linear(vis_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim+vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.attention_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.attention_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.attention_bias = nn.Parameter(torch.zeros(vis_num))
        self.attention_w = nn.Linear(self.vis_dim, 1, bias=False)
        
    def _attention_layer(self, features, hiddens):
        attention_feature = self.attention_vw(features)
        attention_h = self.attention_hw(hiddens).unsqueeze(1)
        attention_full = nn.ReLU()(attention_feature + attention_h + self.attention_bias.view(1, -1, 1))
        attention_out = self.attention_w(attention_full).squeeze(2)
        alpha = nn.Softmax(dim=1)(attention_out).float() # CHECK DIM

        context = torch.sum(features * alpha.unsqueeze(2), dim=1)
        
        return context, alpha
    
    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = to_variable(torch.zeros(batch_size, hidden_dim))
        c0 = to_variable(torch.zeros(batch_size, hidden_dim))
        
        return h0, c0

    def init_hidden_state(self, feature):
        init_h = nn.Linear(self.vis_dim, self.hidden_dim)
        init_c = nn.Linear(self.vis_dim, self.hidden_dim)
        mean_feature = feature.mean(dim=1).data.cpu()
        
        h0 = init_h(mean_feature.data)
        c0 = init_c(mean_feature.data)
        
        h0 = to_variable(h0)
        c0 = to_variable(c0)
        
        return h0, c0
    
    def forward(self, features, captions, lengths):
        batch_size, time_step = captions.data.shape
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out
        
        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(features, 1)
        h0, c0 = self.get_start_states(batch_size)
        
        predicts = to_variable(torch.zeros(batch_size, time_step, vocab_size))
        num_pixels = features.size(1)
        alphas = to_variable(torch.zeros(batch_size, time_step, num_pixels))
        for step in range(time_step):
            if step != 0:
                feas, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])
                alphas[:batch_size, step,:] = alpha
            
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0) if fc_dropout is not None else fc_out(h0))
            predicts[:batch_size, step, :] = outputs
          
        return predicts, alphas
    
    def captioning(self, feature, max_len=13):
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)
        
        cap_ids_lst = []
        alphas_lst = [0.0]
        
        words = embed(to_variable(torch.ones(batch_size, 1).long())).squeeze(1)
        h0, c0 = self.get_start_states(batch_size)
        feas = torch.mean(feature, 1)
        
        for step in range(max_len):
            if step != 0:
                feas, alpha = attend(feature, h0)
                alphas_lst.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            cap_ids_lst.append(predicted.unsqueeze(1))
            words = embed(predicted)
            
        cap_ids = torch.cat(cap_ids_lst, 1)

        return cap_ids.squeeze(), alphas_lst


    def beam_search_captioning(self, feature, vocab, beam_size=3, max_len=13):
        """
        feature   : (batch_size, num_pixles, encoder_dim)
        vocab     :
        beam_size :
        max_len   :
        """

        embed_dim = self.embed_dim
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0) # batch_size = 1
        num_pixels = feature.size(1) # num_pixles = 14*14
        encoder_dim = feature.size(2)
        vocab_size = len(vocab) 

        # expand to the size of beam size
        # shape = (beam_size, num_pixels, encoder_dim)
        feature = feature.expand(beam_size, num_pixels, encoder_dim)

        # store top k previous characters at each step
        k_prev_chars = torch.FloatTensor([[vocab("<start>")]] * beam_size)
        chars = embed(to_variable(k_prev_chars.long())).squeeze(1)

        # store top k sequences
        seqs = torch.cuda.FloatTensor(k_prev_chars.cuda())

        # store top k score of sequence
        top_k_scores = torch.zeros(beam_size, 1).cuda()

        # store top k alpha of sequence
        # (beam_size, 1, img_size, img_size) ?
        seqs_alpha = torch.ones(beam_size, 1, num_pixels).cuda()

        # list of completed sequences
        comp_seq_lst = []
        comp_alpha_lst = []
        comp_score_lst = []

        # initialization
        #h0, c0 = self.init_hidden_state(feature)
        h0, c0 = self.get_start_states(beam_size)
        feas = torch.mean(feature, dim=1)
        alpha = torch.zeros(beam_size, num_pixels).cuda()

        # beam search captioning
        for step in range(max_len):
            if step != 0:
                chars = embed(to_variable(k_prev_chars.long())).squeeze(1)
                feas, alpha = attend(feature.cuda(), h0)
            
            inputs = torch.cat([feas, chars], dim=1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            scores = fc_out(h0)
            scores = F.log_softmax(scores, dim=1)

            # add current score
            scores = top_k_scores.expand_as(scores) + scores

            # NEED FIX: if分岐必要？
            if step == 0:
                top_k_scores, top_k_chars = scores[0].topk(beam_size, 0, True, True)
            else:
                top_k_scores, top_k_chars = scores.view(-1).topk(beam_size, 0, True, True)

            # ???? Convert unrolled indices to actual indices of scores 
            prev_char_inds = top_k_chars / vocab_size
            next_char_inds = top_k_chars % vocab_size

            # add new words and alphas
            seqs = torch.cat([seqs[prev_char_inds], next_char_inds.unsqueeze(1).float()], dim=1)
            seqs_alpha = torch.cat([seqs_alpha[prev_char_inds], alpha[prev_char_inds].unsqueeze(1).float()], dim=1)

            # seach not incomplete sequence
            incomp_inds = [ind for ind, next_char in enumerate(next_char_inds) if next_char != vocab("<end>")]

            # get complete sequence
            comp_inds = list(set(range(len(next_char_inds))) - set(incomp_inds))

            # keep complete sequence
            if len(comp_inds) > 0:
                comp_seq_lst.extend(seqs[comp_inds])
                comp_alpha_lst.extend(seqs_alpha[comp_inds])
                comp_score_lst.extend(top_k_scores[comp_inds])
        
            # decrease beam size
            beam_size -= len(comp_inds)

            # end determination
            if beam_size == 0:
                break
            if step+1 == max_len:
                comp_seq_lst.extend(seqs[incomp_inds])
                comp_alpha_lst.extend(seqs_alpha[incomp_inds])
                comp_score_lst.extend(top_k_scores[incomp_inds])
                break

            # update incomplete sequence
            seqs = seqs[incomp_inds]
            seqs_alpha = seqs_alpha[incomp_inds]
            h0 = h0[prev_char_inds[incomp_inds]]
            c0 = c0[prev_char_inds[incomp_inds]]
            feature = feature[prev_char_inds[incomp_inds]]
            top_k_scores = top_k_scores[incomp_inds].unsqueeze(1)
            k_prev_chars = next_char_inds[incomp_inds].unsqueeze(1)

        # get best sequence and alphas
        # remove <start>
        best_idx = comp_score_lst.index(max(comp_score_lst))
        seq = comp_seq_lst[best_idx][1:]
        alphas = comp_alpha_lst[best_idx][1:]
        
        return seq, alphas



            

