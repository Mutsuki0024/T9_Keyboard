import yaml
import json
import os
import torch
import csv
import Levenshtein

def loadConfig(path: str = 'src/config.yaml') -> dict:
    #Load YAML configuration file.
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def loadVocab(vacabPath:str):
    #load a JSON file mapping token->id, return {token2id, id2token}
    try:
        with open(vacabPath, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
    except FileNotFoundError:
        print(f"File doesn't exist: {vacabPath}")
    id2token = {v:k for k,v in token2id.items()}
    return token2id, id2token

# load config
_cfg = loadConfig()
_cfgData = _cfg['data']
_cfgModel = _cfg['model']
_cfgTrain = _cfg['train']
# load vocabularies
digit2id, id2digit = loadVocab(_cfgData['digitVocab'])
char2id, id2char = loadVocab(_cfgData['charVocab'])


def textToNumber(text):
    #transfer chars to numbers in given text by T9-rule
    cfg = loadConfig()
    try:
        return ''.join(cfg['t9_map'][c] for c in text)
    except:
        return ''


def saveCheckpoint(model, optimizer, scheduler, batch, path: str):
    #Save model, optimizer, scheduler states and current batch
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'batch': batch,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def loadCheckpoint(path: str, model, optimizer=None, scheduler=None):
    #Load checkpoint, restore model (and optional optimizer, scheduler), return epoch.
    device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state'])
    if optimizer and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
    if scheduler and 'scheduler_state' in state:
        scheduler.load_state_dict(state['scheduler_state'])
        
    return state.get('batch',None)


def compute_cer(pred: str, ref: str) -> float:
    """
    Compute Character Error Rate (CER) = edit_distance / len(ref).
    Uses Levenshtein distance on characters.
    """
    if len(ref) == 0:
        return float('inf') if len(pred) > 0 else 0.0
    dist = Levenshtein.distance(pred, ref)
    return dist / len(ref)


def compute_wer(pred: str, ref: str) -> float:
    """
    Compute Word Error Rate (WER) = edit_distance on word sequences / num_words(ref).
    """
    r_words = ref.split()
    p_words = pred.split()
    if len(r_words) == 0:
        return float('inf') if len(p_words) > 0 else 0.0
    # compute distance between word lists by joining with a sentinel
    r_join = '\x1f'.join(r_words)
    p_join = '\x1f'.join(p_words)
    dist = Levenshtein.distance(p_join, r_join)
    return dist / len(r_words)


def splitWords(seqIDs, splitWord):
    #Split a 1D list/iterable of token IDs into list of word tuples,
    words = []
    current = []
    for id in seqIDs:
        if id == splitWord:
            if current:
                words.append(tuple(current))
                current = []
        else:
            current.append(int(id))
    if current: #the last word
        words.append(tuple(current))
    return words

def compute_char_accuracy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    #Compute number of correct characters and total characters.
    
    valid = mask.bool()
    correct = (preds == labels) & valid
    return int(correct.sum().item()), int(valid.sum().item())  #return num of corect, total num



def compute_word_accuracy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    #Compute number of correct words and total words.

    batchSize = preds.size(0)
    totalWords = 0
    correctWords = 0

    charSplitID = char2id.get(' ')
    for i in range(batchSize):
        validPositions = mask[i].bool()
        predSeq = preds[i][validPositions].tolist()
        labelSeq = labels[i][validPositions].tolist()
        predWords = splitWords(predSeq, charSplitID)
        labelWords = splitWords(labelSeq, charSplitID)
        totalWords += len(labelWords)
        # compare word by position
        for pw, lw in zip(predWords, labelWords):
            if pw == lw:
                correctWords += 1
    return correctWords, totalWords

def compute_sentence_accuracy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    #Compute number of fully correct sentences and total sentences.
    batch_size = preds.size(0)
    total = batch_size
    correct = 0
    for i in range(batch_size):
        valid_positions = mask[i].bool()
        if torch.equal(preds[i][valid_positions], labels[i][valid_positions]):
            correct += 1
    return correct, total

def writeLogs(batch, avgLoss, charAcc, wordAcc, sentAcc):
    outputDir = os.path.join(_cfgTrain['logsDir'], 'train_logs.csv')
    needWriteHeader = not os.path.exists(outputDir) or os.stat(outputDir).st_size==0
    with open(outputDir, 'a', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        if needWriteHeader:
            writer.writerow(['batch','avgLoss','charAcc','wordAcc','sentAcc'])
        writer.writerow([format(batch,"<6"), format(avgLoss, ".5f"), format(charAcc, ".5f"), format(wordAcc, ".5f"), format(sentAcc, ".5f")])
        