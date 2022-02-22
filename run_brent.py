#!/usr/bin/env python

"""
Train a DPDP AE-RNN and perform word segmentation on the Brent corpus.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from scipy.stats import gamma
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn

from dpdp_aernn import datasets, models, viterbi
from utils import eval_segmentation


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0]  #, add_help=False
        )
    parser.add_argument(
        "--input_dropout", type=float, default=0.0,
        help="probability with which an input embedding is dropped "
        "(default: %s(default))"
        )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def get_segmented_sentence(ids, boundaries, id_to_symbol, join_char=""):
    output = ""
    cur_word = []
    for i_symbol, boundary in enumerate(boundaries):
        cur_word.append(id_to_symbol[ids[i_symbol]])
        if boundary:
            output += join_char.join(cur_word)
            output += " "
            cur_word = []
    return output.strip()


#-----------------------------------------------------------------------------#
#                          DURATION PENALTY FUNCTIONS                         #
#-----------------------------------------------------------------------------#

histogram = np.array([
    0., 0.051637, 0.36365634, 0.35984765, 0.1537391,
    0.04632681, 0.01662638, 0.00644547, 0.00131839, 0.00040284,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001
    ])
histogram = histogram/np.sum(histogram)
def neg_log_hist(dur):
    return -np.log(0 if dur >= len(histogram) else histogram[dur])

def neg_log_psuedo_geometric(dur):
    return -(dur - 1)

# Cache Gamma
shape, loc, scale = (7, 0, 0.4)
gamma_cache = []
for dur in range(50):
    gamma_cache.append(gamma.pdf(dur, shape, loc, scale))
gamma_cache = np.array(gamma_cache)
def neg_log_gamma(dur):
    if dur < 50:
        return -np.log(gamma_cache[dur])
    else:
        return -np.log(gamma.pdf(dur, shape, loc, scale))


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # DATA

    # Load data
    fn = Path("data")/"br-phono.txt"
    print("Reading:", fn)
    sentences_ref = []
    with open(fn) as f:
        for line in f:
            sentences_ref.append(line.strip())     
    print("No. sentences:", len(sentences_ref))
    train_sentences_ref = sentences_ref[:]  # to-do
    val_sentences_ref = sentences_ref[:1000]
    test_sentences_ref = sentences_ref[8000:]

    print("\nExample training sentence reference:")
    print(train_sentences_ref[0])

    # Vocabulary
    PAD_SYMBOL = "<pad>"
    SOS_SYMBOL = "<s>"    # start of sentence
    EOS_SYMBOL = "</s>"   # end of sentence
    symbols = set()
    for sentence in sentences_ref:
        for char in sentence:
            symbols.add(char)
    SYMBOLS = [PAD_SYMBOL, SOS_SYMBOL, EOS_SYMBOL] + (sorted(list(symbols)))
    symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
    id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}

    def text_to_id(text, add_sos_eos=False):
        """
        Convert text to a list of symbol IDs.

        Sentence start and end symbols can be added by setting `add_sos_eos`.
        """
        symbol_ids = [symbol_to_id[t] for t in text]
        if add_sos_eos:
            return ([
                symbol_to_id[SOS_SYMBOL]] + symbol_ids +
                [symbol_to_id[EOS_SYMBOL]
                ])
        else:
            return symbol_ids
    print(text_to_id(train_sentences_ref[0]))
    print(
        [id_to_symbol[i] for i in  text_to_id(train_sentences_ref[0])]
        )
    print()

    cur_train_sentences = train_sentences_ref
    cur_val_sentences = val_sentences_ref[:100]
    cur_train_sentences = ["".join(i.split(" ")) for i in cur_train_sentences]


    # MODEL

    n_symbols = len(SYMBOLS)
    symbol_embedding_dim = 25
    hidden_dim = 200
    embedding_dim = 25
    teacher_forcing_ratio = 0.5  # 1.0
    n_encoder_layers = 3  # 2  # 1  # 3  # 10
    n_decoder_layers = 1  # 2  # 1
    batch_size = 32  # 32
    learning_rate = 0.001
    input_dropout = args.input_dropout  
    dropout = 0.0
    n_epochs_max = 5

    encoder = models.Encoder(
        n_symbols=n_symbols,
        symbol_embedding_dim=symbol_embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_encoder_layers,
        dropout=dropout,
        input_dropout=input_dropout,
        )
    decoder = models.Decoder1(
        n_symbols=n_symbols,
        symbol_embedding_dim=symbol_embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_decoder_layers,
        sos_id = symbol_to_id[SOS_SYMBOL],
        teacher_forcing_ratio=teacher_forcing_ratio,
        dropout=dropout
        )
    # decoder = models.Decoder2(
    #     n_symbols=n_symbols,
    #     hidden_dim=hidden_dim,
    #     embedding_dim=embedding_dim,
    #     n_layers=n_decoder_layers,
    #     )
    model = models.EncoderDecoder(encoder, decoder)


    # TRAINING

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training data
    train_dataset = datasets.WordDataset(cur_train_sentences, text_to_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=datasets.pad_collate
        )

    # Validation data
    val_dataset = datasets.WordDataset(cur_val_sentences, text_to_id)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=datasets.pad_collate
        )

    # Loss
    criterion = nn.NLLLoss(
        reduction="sum", ignore_index=symbol_to_id[PAD_SYMBOL]
        )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training AE-RNN:")
    for i_epoch in range(n_epochs_max):

        # Training
        model.train()
        train_losses = []
        for i_batch, (data, data_lengths) in enumerate(tqdm(train_loader)):
            optimiser.zero_grad()
            data = data.to(device)       
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )

            loss = criterion(
                decoder_output.contiguous().view(-1, decoder_output.size(-1)),
                data.contiguous().view(-1)
                )
            loss /= len(data_lengths)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i_batch, (data, data_lengths) in enumerate(val_loader):
                data = data.to(device)            
                encoder_embedding, decoder_output = model(
                    data, data_lengths, data, data_lengths
                    )

                loss = criterion(
                    decoder_output.contiguous().view(-1,
                    decoder_output.size(-1)), data.contiguous().view(-1)
                    )
                loss /= len(data_lengths)
                val_losses.append(loss.item())
        
        print(
            "Epoch {}, train loss: {:.3f}, val loss: {:.3f}".format(
            i_epoch,
            np.mean(train_losses),
            np.mean(val_losses))
            )
        sys.stdout.flush()
    print()

    # Examples without segmentation

    # Apply to validation data
    print("Examples:")
    model.eval()
    with torch.no_grad():
        for i_batch, (data, data_lengths) in enumerate(val_loader):
            data = data.to(device)
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )
            
            y, log_probs = model.decoder.greedy_decode(
                encoder_embedding,
                max_length=20,
                )
            x = data.cpu().numpy()
            
            for i_input in range(y.shape[0]):
                # Only print up to EOS symbol
                input_symbols = []
                for i in x[i_input]:
                    if (i == symbol_to_id[EOS_SYMBOL] or i ==
                            symbol_to_id[PAD_SYMBOL]):
                        break
                    input_symbols.append(id_to_symbol[i])
                output_symbols = []
                for i in y[i_input]:
                    if (i == symbol_to_id[EOS_SYMBOL] or i ==
                            symbol_to_id[PAD_SYMBOL]):
                        break
                    output_symbols.append(id_to_symbol[i])

                print("Input: ", "".join(input_symbols))
                print("Output:", "".join(output_symbols))
                
                if i_input == 5:
                    break
            
            break
    print()


    # SEGMENTATION

    # Embed segments

    # Data
    sentences = val_sentences_ref
    # sentences = test_sentences_ref
    # sentences = train_sentences_ref
    interval_dataset = datasets.SentenceIntervalDataset(sentences, text_to_id)
    segment_loader = DataLoader(
        interval_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=datasets.pad_collate,
        drop_last=False
        )

    # Apply model to data
    model.decoder.teacher_forcing_ratio = 1.0
    model.eval()
    rnn_losses = []
    lengths = []
    print("Embedding segments:")
    with torch.no_grad():
        for i_batch, (data, data_lengths) in enumerate(tqdm(segment_loader)):
            data = data.to(device)
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )

            for i_item in range(data.shape[0]):
                item_loss = criterion(
                    decoder_output[i_item].contiguous().view(-1,
                    decoder_output[i_item].size(-1)),
                    data[i_item].contiguous().view(-1)
                    )
                rnn_losses.append(item_loss)
                lengths.append(data_lengths[i_item])
    print()

    # Segment
    dur_weight = 1.5
    i_item = 0
    predicted_boundaries = []
    reference_boundaries = []
    losses = []
    cur_segmented_sentences = []
    print("Segmenting:")
    for i_sentence, intervals in enumerate(tqdm(interval_dataset.intervals)):
        
        # Costs for segment intervals
        costs = np.inf*np.ones(len(intervals))
        i_eos = intervals[-1][-1]
        for i_seg, interval in enumerate(intervals):
            if interval is None:
                continue
            i_start, i_end = interval
            dur = i_end - i_start
            assert dur == lengths[i_item]
            eos = (i_end == i_eos)  # end-of-sequence
            
            # Gamma
            costs[i_seg] = (
                rnn_losses[i_item]
                + dur_weight*neg_log_gamma(dur)
                + np.log(np.sum(gamma_cache**dur_weight))
                )

            # # Histogram
            # costs[i_seg] = (
            #     rnn_losses[i_item]
            #     + dur_weight*(neg_log_hist(dur))
            #     + np.log(np.sum(histogram**dur_weight))
            #     )
            
            # Sequence boundary
            alpha = 0.1  # 0.9
            if eos:
                costs[i_seg] += -np.log(alpha)
            else:
                # costs[i_seg] += -np.log(1 - alpha)
                K = 1
                costs[i_seg] += -np.log((1 - alpha)/K)
            
            i_item += 1
        
        # Viterbi segmentation
        n_frames = len(interval_dataset.sentences[i_sentence])
        summed_cost, boundaries = viterbi.custom_viterbi(costs, n_frames)
        losses.append(summed_cost)
        
        reference_sentence = sentences[i_sentence]
        segmented_sentence = get_segmented_sentence(
                interval_dataset.sentences[i_sentence],
                boundaries, id_to_symbol
                )
        cur_segmented_sentences.append(segmented_sentence)
        # Print examples of the first few sentences
        if i_sentence < 10:
            print(reference_sentence)
            print(segmented_sentence)
            # print()
        
        predicted_boundaries.append(boundaries)
        reference_boundaries.append(
            datasets.sentence_to_boundaries(reference_sentence)
            )

    print("NLL: {:.4f}\n".format(np.sum(losses)))


    # EVALUATION

    p, r, f  = eval_segmentation.score_boundaries(
        reference_boundaries, predicted_boundaries
        )
    print("-"*(79 - 4))
    print("Word boundaries:")
    print("Precision: {:.4f}%".format(p*100))
    print("Recall: {:.4f}%".format(r*100))
    print("F-score: {:.4f}%".format(f*100))
    print("OS: {:.4f}%".format(eval_segmentation.get_os(p, r)*100))
    print("-"*(79 - 4))

    p, r, f = eval_segmentation.score_word_token_boundaries(
        reference_boundaries, predicted_boundaries
        )
    print("Word token boundaries:")
    print("Precision: {:.4f}%".format(p*100))
    print("Recall: {:.4f}%".format(r*100))
    print("F-score: {:.4f}%".format(f*100))
    print("OS: {:.4f}%".format(eval_segmentation.get_os(p, r)*100))
    print("-"*(79 - 4))


if __name__ == "__main__":
    main()
