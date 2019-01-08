from __future__ import division, print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import os
import utils, models


parser = argparse.ArgumentParser()
parser.add_argument("--treebank_path", type=str)
parser.add_argument("--optim", type=str, default='adam', choices=["sgd","adam","adagrad","rmsprop"])
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--mlp_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--jackknife", action='store_true')
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--langs", type=str, default="uk",
help="Languages separated by delimiter '/' with last language being target language")
parser.add_argument("--tgt_size", type=int, default=None)
parser.add_argument("--tgt_frac", type=float, default=None)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--model_name", type=str, default="model_pos")
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--model_type", type=str, default="baseline", choices=["universal","joint","mono","specific","baseline"])
parser.add_argument("--sum_word_char", action='store_true')
parser.add_argument("--sent_attn", action='store_true')
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--test", type=str, default=None, choices=["dev_set", "test_set"])
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
print(args)

# Set seed
torch.manual_seed(args.seed)

langs = args.langs.split("/")
args.model_name = args.model_path + args.model_type + "".join(["_" + l for l in langs])
if args.sum_word_char:
    args.model_name += "_wc-sum"
if args.sent_attn:
    args.model_name += "_sent-attn"
if args.tgt_size:
    args.model_name += "-" + str(args.tgt_size)
if args.tgt_frac:
    args.model_name += "-" + str(args.tgt_frac)
if args.jackknife and args.fold!=-1:
    args.model_name += "-" + str(args.fold)

print("Reading training data...")

training_data_langwise, train_tgt_labels = utils.read_unimorph(args.treebank_path, langs, tgt_size=args.tgt_size, train_or_dev="train")
training_data = []

for l in langs:
    training_data += training_data_langwise[l]

labels_to_ix = train_tgt_labels
labels_to_ix = {k: v for v, k in enumerate(train_tgt_labels)}
print("Number of unique tagsets: %d" % len(labels_to_ix))

# Jackknifing
if args.jackknife:
    train_datasets, dev_datasets = utils.jackknife(training_data)

dev_data_langwise, dev_tgt_labels = utils.read_unimorph(args.treebank_path, [langs[-1]], train_or_dev="dev", test=True)
dev_data = dev_data_langwise[langs[-1]]

dev_or_test = "dev" if args.test == "dev_set" else "test"

if args.test:
    test_lang = langs[-1]
    test_data_langwise, test_tgt_labels = utils.read_unimorph(args.treebank_path, [test_lang], train_or_dev=dev_or_test, test=True)
    test_data = test_data_langwise[test_lang]
    
word_to_ix = {}
char_to_ix = {}
word_freq = {}
for sent, _ in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        if word_to_ix[word] not in word_freq:
            word_freq[word_to_ix[word]] = 1
        else:
            word_freq[word_to_ix[word]] += 1
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)



word_to_ix["UNK"] = len(word_to_ix)
char_to_ix["UNK"] = len(char_to_ix)
labels_to_ix["UNK"] = len(labels_to_ix)
ix_to_labels = {v: k for k, v in labels_to_ix.items()}

if args.model_type=='universal':
    for lang in langs:
        char_to_ix[lang] = len(char_to_ix)

# training_data_langwise.sort(key=la:mbda x: -len(x[0]))
# test_data.sort(key=lambda x: -len(x[0]))
# train_order = [x*args.batch_size for x in range(int((len(training_data_langwise)-1)/args.batch_size + 1))]
# test_order = [x*args.batch_size for x in range(int((len(test_data)-1)/args.batch_size + 1))]

def train(k, training_data_jack, dev_data_jack):

    if not os.path.isfile(args.model_name) or args.continue_train:
        if args.continue_train:
            print("Loading tagger model from " + args.model_name + "...")
            tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
            if args.gpu:
                tagger_model = tagger_model.cuda()

        else:
            tagger_model = models.BiLSTMTagger(args, word_freq, langs, len(char_to_ix), len(word_to_ix), len(labels_to_ix))
            if args.gpu:
                tagger_model = tagger_model.cuda()

        loss_function = nn.NLLLoss()

        if args.optim=="sgd":
            optimizer = optim.SGD(tagger_model.parameters(), lr=0.1)
        elif args.optim=="adam":
            optimizer = optim.Adam(tagger_model.parameters())
        elif args.optim=="adagrad":
            optimizer = optim.Adagrad(tagger_model.parameters())
        elif args.optim=="rmsprop":
            optimizer = optim.RMSprop(tagger_model.parameters())

        print("Training tagger model...")
        patience_counter = 0
        prev_avg_tok_accuracy = 0
        for epoch in range(args.epochs):
            accuracies = []
            sent = 0
            tokens = 0
            cum_loss = 0
            correct = 0
            print("Starting epoch %d .." %epoch)
            for lang in langs:
                lang_id = []
                if args.model_type=="universal":
                    lang_id = [lang]
                for sentence, morph in training_data_jack:
                    sent += 1

                    if sent%100==0:

                        print("[Epoch %d] \
                            Sentence %d/%d, \
                            Tokens %d \
                            Cum_Loss: %f \
                            Average Accuracy: %f"
                            % (epoch, sent, len(training_data_jack), tokens,
                                cum_loss/tokens, correct/tokens))

                    tagger_model.zero_grad()
                    sent_in = []
                    tokens += len(sentence)

                    for word in sentence:
                        s_appended_word  = lang_id + [c for c in word] + lang_id
                        word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
                        # targets = utils.prepare_sequence(s_appended_word[1:], char_to_ix, args.gpu)
                        sent_in.append(word_in)

                    # sent_in = torch.stack(sent_in)
                    tagger_model.char_hidden = tagger_model.init_hidden()
                    tagger_model.hidden = tagger_model.init_hidden()

                    targets = utils.prepare_sequence(morph, labels_to_ix, args.gpu)

                    if args.sum_word_char:
                        word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
                    else:
                        word_seq = None

                    if args.model_type=="specific" or args.model_type=="joint":
                        tag_scores = tagger_model(sent_in, word_idxs=word_seq, lang=lang)
                    else:
                        tag_scores = tagger_model(sent_in, word_idxs=word_seq)

                    values, indices = torch.max(tag_scores, 1)
                    out_tags = indices.cpu().data.numpy().flatten()
                    correct += np.count_nonzero(out_tags==targets.cpu().data.numpy())
                    loss = loss_function(tag_scores, targets)
                    cum_loss += loss.cpu().data[0]
                    loss.backward()
                    optimizer.step()

            print("Loss: %f" % loss.cpu().data.numpy())
            print("Accuracy: %f" %(correct/tokens))
            print("Saving model..")
            torch.save(tagger_model, args.model_name)
            #print("Evaluating on dev set...")
            #avg_tok_accuracy = eval(tagger_model, curEpoch=epoch)

            # Early Stopping
            #if avg_tok_accuracy <= prev_avg_tok_accuracy:
            #    patience_counter += 1
            #    if patience_counter==args.patience:
            #        print("Model hasn't improved on dev set for %d epochs. Stopping Training." % patience_counter)
            #        break

            #prev_avg_tok_accuracy = avg_tok_accuracy
    else:
        print("Loading tagger model from " + args.model_name + "...")
        tagger_model = torch.load(args.model_name, map_location=lambda storage, loc: storage)
        if args.gpu:
            tagger_model = tagger_model.cuda()

    if args.test:
        avg_tok_accuracy = eval(tagger_model, args.fold, dev_or_test=dev_or_test)
   
    return tagger_model

def eval(tagger_model, k, dev_or_test="dev"):

    if k==-1:
        eval_data = dev_data if dev_or_test=="dev" else test_data
    else:
        eval_data = dev_datasets[k]
    correct = 0
    toks = 0
    hypTags = []
    goldTags = []
    all_out_tags = np.array([])
    all_targets = np.array([])
    logProbs = []
    print("Starting evaluation on %s set... (%d sentences)" % (dev_or_test, len(eval_data)))
    lang_id = []
    if args.model_type=="universal":
        lang_id = [lang]
    sentCount = 0
    for sentence, morph in eval_data:
        tagger_model.zero_grad()
        tagger_model.char_hidden = tagger_model.init_hidden()
        tagger_model.hidden = tagger_model.init_hidden()
        sent_in = []
        sentCount += 1
        for word in sentence:
            s_appended_word  = lang_id + [c for c in word] + lang_id
            word_in = utils.prepare_sequence(s_appended_word, char_to_ix, args.gpu)
            sent_in.append(word_in)

        #targets = utils.prepare_sequence(morph, labels_to_ix, args.gpu)
        if args.sum_word_char:
            word_seq = utils.prepare_sequence(sentence, word_to_ix, args.gpu)
        else:
            word_seq = None

        if args.model_type=="specific":
            tag_scores = tagger_model(sent_in, word_idxs=word_seq, lang=langs[-1], test=True)
        else:
            tag_scores = tagger_model(sent_in, word_idxs=word_seq, test=True)

        tag_scores = tag_scores[:, :-1]

        #values, indices = torch.topk(tag_scores, k=100, dim=1)
        values, indices = torch.max(tag_scores, 1)
        out_tags = indices.cpu().data.numpy()
        #for i in range(out_tags.shape[0]):
        #    hypTags.append([utils.unfreeze_dict(ix_to_labels[idx]) for idx in out_tags[i]])
        hypTags.append([ix_to_labels[idx] for idx in out_tags])
        scores = values.cpu().data.numpy()
        #logProbs += [list(scores[i]) for i in range(scores.shape[0])]
        #all_out_tags = np.append(all_out_tags, out_tags)
        goldTags.append(morph)
        #targets = targets.cpu().data.numpy()
        #correct += np.count_nonzero(out_tags==targets)
        #print(out_tags)
        #correct += np.count_nonzero(np.array([ix_to_labels[idx] for idx in out_tags])==np.array(morph))
        toks += len(sentence)

    avg_tok_accuracy = correct / toks

    prefix = args.model_type + "_"
    if args.sum_word_char:
        prefix += "_wc-sum"

    prefix += "-".join([l for l in langs]) + "_" + dev_or_test

    if args.sent_attn:
        prefix += "-sent_attn"

    if args.tgt_size:
        prefix += "_" + str(args.tgt_size)

    write = True
    folds = 10
    dev_size = (int)(len(training_data)/folds) if args.jackknife else None
    if write:
        utils.write_unimorph(args.treebank_path, hypTags, logProbs, sentCount, k, dev_or_test=dev_or_test, dev_size=dev_size)

    return avg_tok_accuracy

if __name__=="__main__":
    if args.jackknife:
        folds = 10
        dev_size = (int)(len(training_data)/folds)
        k = args.fold
        print("Fold %d" %k)
        tagger_model = train(k, train_datasets[k], dev_datasets[k])
    else:
        tagger_model = train(args.fold, training_data, dev_data)
        #eval(tagger_model, args.fold, dev_or_test="test")
