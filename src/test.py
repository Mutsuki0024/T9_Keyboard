import utils
import torch

# load config
cfg = utils.loadConfig()
cfgData = cfg['data']
cfgModel = cfg['model']
cfgTrain = cfg['train']

# load vocabularies
digit2id, id2digit = utils.loadVocab(cfgData['digitVocab'])
char2id, id2char = utils.loadVocab(cfgData['charVocab'])

text = 'this is a test'
pred = 'thas is a tset'
textIDs = torch.tensor(
            [char2id.get(ch, 0) for ch in text],
            dtype=torch.long
        )
predIDs = torch.tensor(
            [char2id.get(ch, 0) for ch in pred],
            dtype=torch.long
        )
mask = torch.tensor(
    [True]*len(text)
)
print(textIDs)
print(predIDs)
print(utils.splitWords(textIDs.tolist(),char2id.get(' ')))