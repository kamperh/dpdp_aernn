"""
Dataset classes.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
import torch


class WordDataset(Dataset):
    """
    Isolated word dataset.

    Parameters
    ----------
    sentences : list of str
        In each sentence, words are assumed to be separated by spaces.
    text_to_id_func : function
        Should map a given string to a list of IDs.
    """
    
    def __init__(self, sentences, text_to_id_func, n_symbols_max=None):
        self.words = []
        self.lengths = []
        for sentence in sentences:
            for word in sentence.split(" "):
                ids = text_to_id_func(word)
                if n_symbols_max is not None:
                    ids = ids[:n_symbols_max]
                    # # Temp
                    # if len(ids) > n_symbols_max:
                    #     i_start = np.random.randint(len(ids) - n_symbols_max)
                    #     ids = ids[i_start:i_start+n_symbols_max]

                self.words.append(ids)
                self.lengths.append(len(ids))

    def sort_key(self, i_item):
        return self.lengths[i_item]
                
    def __len__(self):
        return(len(self.words))

    def __getitem__(self, i_item):
        return torch.LongTensor(self.words[i_item])


class SentenceIntervalDataset(Dataset):
    """Sentences are split into separate fragmented intervals."""
    
    def __init__(self, sentences, text_to_id_func, join_char="",
            n_words_max=None, n_symbols_max=15):
        self.sentences = []
        self.intervals = []
        self.fragments = []
        self.lengths = []
        
        for sentence in sentences:
            
            # Remove spaces and get IDs
            tmp = sentence.split(" ")
            if n_words_max is None:
                tmp = join_char.join(tmp)
            else:
                tmp = tmp[:n_words_max]
                tmp = join_char.join(tmp)
            ids = text_to_id_func(tmp)
            self.sentences.append(ids)
            
            # Get intervals
            intervals = get_segment_intervals(len(ids), n_symbols_max)
            self.intervals.append(intervals)
            for i_seg, interval in enumerate(intervals):
                if interval is None:
                    continue
                i_start, i_end = interval
                dur = i_end - i_start
                self.fragments.append(ids[i_start:i_end])
                self.lengths.append(dur)
               
    def __len__(self):
        return(len(self.fragments))

    def __getitem__(self, i_item):
        return torch.LongTensor(self.fragments[i_item])


def sentence_to_boundaries(sentence):
    boundaries = []
    for char in sentence:
        if char == " ":
            boundaries[-1] = True
        else:
            boundaries.append(False)
    boundaries[-1] = True
    return np.array(boundaries)


def get_segment_intervals(n_total, n_max_frames):
    indices = [None]*int((n_total**2 + n_total)/2)
    for cur_start in range(n_total):
        for cur_end in range(cur_start, min(n_total, cur_start +
                n_max_frames)):
            cur_end += 1
            t = cur_end
            i = int(t*(t - 1)/2)
            indices[i + cur_start] = (cur_start, cur_end)
    return indices


def pad_collate(batch, padding_value=0):
    texts = batch
    text_lengths = [len(text) for text in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=padding_value)
    return texts, text_lengths

