from matplotlib import pyplot as plt

if False:
    import os 
    os.system('ls -ltrh')
    os.system('ls -ltrh former')
    os.system('pwd')

    from pathlib import Path

    for p in Path('.').glob('**'): 
        print(p)

from _context import former

from former import util

from util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, sys, math, gzip
from tqdm.auto import tqdm #MK
#from tqdm.autonotebook import tqdm #MK

import os #MK

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

def token_ids_to_text(token_ids):
    return ' '.join([TEXT.vocab.itos[token_id] for token_id in token_ids])

def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging
    #from ipdb import set_trace; set_trace()
    # load the IMDB data
    if arg.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)
        #train = train[:500]
        #test = test[:100]

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())

    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    for e in range(arg.num_epochs):

        print(f'\n epoch {e}')
        model.train(True)

        for i_, batch in enumerate(tqdm(train_iter)): #MK

            if False and i_ > 5:
                continue
                
            opt.zero_grad()

            input = batch.text[0]
            label = batch.label - 1

            if input.size(1) > mx:
                input = input[:, :mx]
            out, attentions = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            seen += input.size(0)
            tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0

            for i, batch in enumerate(test_iter):
                
                if False and i > 3:
                    break

                input = batch.text[0]
                label = batch.label - 1

                if input.size(1) > mx:
                    input = input[:, :mx]
                out, attentions = model(input)
                out = out.argmax(dim=1)
                if i == 5:
                    #print('attentions', attentions[1].size(), len(attentions))
                    #print('---')
                    
                    batch_size = len(batch.text[0])
                    #print('batch size', batch_size)
                    layers = arg.depth
                    #print('layers', layers)
                    heads = arg.num_heads
                    #print('heads', heads)
                    
                    
                    #fig.tight_layout(pad=0, w_pad=0, h_pad=0)
                    
                    for j in range(batch_size)[:1]:
                        
                        print('instance in batch:', j)
                        
                        fig = plt.figure(figsize=(20, 20)) 
    
                        #attention_matrices = attentions[0][j] #[0] # layer, batch, head
    
                        words = token_ids_to_text(batch.text[0][j])
                        print('label', label[j].item())
                        print(words)
                        words = [f'{w:20s}' for w in words.split()]
                        #from ipdb import set_trace; set_trace()
                        for l in range(layers):
                            for h in range(heads):
                                # layers as rows, 
                                # heads as columns
                                # FIXME: Which layers go on top?
                                
                                index = l * heads + h + 1
                                #print('layer', l, 'head', h, 'index', index)
                                
                                #from ipdb import set_trace; set_trace()
                                
                                ax = fig.add_subplot(layers, heads, index)
                                
                                ax.set_xticks(range(len(words)))
                                ax.set_xticklabels(words)
                                ax.tick_params(axis='x', rotation=90)

                                ax.set_yticks(range(len(words)))
                                ax.set_yticklabels(words)
                                

                                if l != layers-1: # not last row?
                                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                                if h != 0: # not first column?
                                    #ax.set_yticklabels(len(words)*[''])
                                    ax.tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False)
                                    
                                ax.imshow(attentions[l][j][h].cpu())
                                
                                plt.subplots_adjust(wspace=0, hspace=0)
                                #plt.show(img)
                        
                        
                        #img.show()
                        display(fig)
                    #plt.close()
                        #plt.show()
                    print('..')
                    #from ipdb import set_trace; set_trace()
                    
                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            tbw.add_scalar('classification/test-loss', float(loss.item()), e)
            
    # Saving model
    print('Saving model to ', os.environ.get('SM_MODEL_DIR'))
    torch.save(model, os.environ.get('SM_MODEL_DIR'))
    os.system(f'ls {os.environ.get("SM_MODEL_DIR")}')
            
def get_arg_parser():
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
    
    return parser

if __name__ == "__main__":

    options = get_arg_parser().parse_args()

    print('OPTIONS ', options)

    go(options)
