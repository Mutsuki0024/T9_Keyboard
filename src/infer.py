import argparse
import yaml
import torch
import utils
from model import T9Labeler

_cfg = utils.loadConfig()
_cfgData = _cfg['data']
_cfgModel = _cfg['model']
_cfgTrain = _cfg['train']


# load vocabularies
_digit2id, _ = utils.loadVocab(_cfgData['digitVocab'])
_char2id, _id2char = utils.loadVocab(_cfgData['charVocab'])

# load T9 map
t9Map = _cfg['t9_map']

#get device
_device = _cfgTrain['device']


def buildModel(ckptPath):
    model = T9Labeler(
        digitVocabSize=len(_digit2id),
        charVocabSize=len(_char2id),
        dModel=_cfgModel['dModel'],
        nhead=_cfgModel['nhead'],
        numLayers=_cfgModel['numLayers'],
        dimFeedForward=_cfgModel['dimFeedforward'],
        dropout=_cfgModel['dropout'],
        maxLen=_cfgModel['maxLen']
    ).to(_device)

    #load model parameter
    state = utils.loadCheckpoint(ckptPath,model)
    model.eval()
    return model
def infer(model,digits):
    if not model:
        model = buildModel()
        print('default model is used.')
    
    inputIDs = torch.tensor([_digit2id[d] for d in digits],dtype=torch.long,device=_device).unsqueeze(0)
    attentionMask = torch.ones_like(inputIDs, dtype=torch.bool, device=_device)
    
    #infer
    with torch.no_grad():
        logits = model(inputIDs, attentionMask)
        preds = logits.argmax(dim=-1)[0].tolist()
        chars = ''.join([_id2char[id] for id in preds])
    return chars
    


def main():
    #initialize model
    parser = argparse.ArgumentParser(description='T9 Infer Script')
    parser.add_argument("--ckpt", type=str, required=False,
                        help="Path to model checkpoint .pt file")
    args = parser.parse_args()

    modelPath = _cfgModel['bestModel']
    if args.ckpt:
        modelPath = args.ckpt
        
    model = buildModel(modelPath)
    
    print("Enter text to tranfer to digits and predict by model. Empty input to exit.")
    while True:
        text = input("Text>>").strip()
        if not text:
            break
        digits = ''.join([t9Map[ch] for ch in text])
        print(f"Input digits: {digits}")
        output = infer(model,digits)
        print(f"Output: {output}")

if __name__ == "__main__":
    main()