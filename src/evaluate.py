import os
import math
import torch
import utils
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import T9Dataset
from model import T9Labeler

# load config
_cfg = utils.loadConfig()
_cfgData = _cfg['data']
_cfgModel = _cfg['model']
_cfgTrain = _cfg['train']

# load vocabularies
_digit2id, _id2digit = utils.loadVocab(_cfgData['digitVocab'])
_char2id, _id2char = utils.loadVocab(_cfgData['charVocab'])



def eval(model=_cfgModel['bestModel'],device=torch.device('cpu')):
    validSet = T9Dataset(
        dataDir=_cfgData['validDataDir'],
        digitVocabPath=_cfgData['digitVocab'],
        charVocabPath=_cfgData['charVocab'],
        csvDelimiter=_cfgData['csvDelimiter'],
        sentenceCheck=False,
        digitCol=_cfgData['digitCol'],
        textCol=_cfgData['textCol'],
        hasHeader=_cfgData['hasHeader']
    )
    validLoader = DataLoader(
        validSet,
        batch_size=_cfgTrain["batchSize"],
        collate_fn=validSet.collate_fn,
        num_workers=_cfgTrain["numWorkers"]
    )
    
    model.eval()
    
    totalChars = totalCharsCorrect = 0
    totalWords = totalWordsCorrect = 0
    totalSents = totalSentsCorrect = 0
    with torch.no_grad():
        for batch in validLoader:
            inputIDs = batch["inputIDs"].to(device)
            labels = batch["labelIDs"].to(device)
            attention_mask = batch["attentionMask"].to(device)
            logits = model(inputIDs, attention_mask)
            preds = logits.argmax(dim=-1)
            b_char_corr, b_char_tot = utils.compute_char_accuracy(preds, labels, attention_mask)
            w_corr, w_tot = utils.compute_word_accuracy(preds, labels, attention_mask)
            s_corr, s_tot = utils.compute_sentence_accuracy(preds, labels, attention_mask)
            totalCharsCorrect += b_char_corr
            totalChars += b_char_tot
            totalWordsCorrect += w_corr
            totalWords += w_tot
            totalSentsCorrect += s_corr
            totalSents += s_tot
    char_acc = totalCharsCorrect / totalChars
    word_acc = totalWordsCorrect / totalWords
    sent_acc = totalSentsCorrect / totalSents
    model.train()
    return char_acc, word_acc, sent_acc
    