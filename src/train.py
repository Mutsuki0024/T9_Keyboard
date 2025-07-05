import os
import math
import torch
import utils
import torch.nn as nn
from evaluate import eval
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import T9Dataset
from model import T9Labeler

def train():
    # load config
    cfg = utils.loadConfig()
    cfgData = cfg['data']
    cfgModel = cfg['model']
    cfgTrain = cfg['train']
    
    # load vocabularies
    digit2id, id2digit = utils.loadVocab(cfgData['digitVocab'])
    char2id, id2char = utils.loadVocab(cfgData['charVocab'])
    
    # prepare dataset
    dataset = T9Dataset(
        dataDir=cfgData['trainDataDir'],
        digitVocabPath=cfgData['digitVocab'],
        charVocabPath=cfgData['charVocab'],
        csvDelimiter=cfgData['csvDelimiter'],
        sentenceCheck=False,  #data has been cleaned and filtered
        digitCol=cfgData['digitCol'],
        textCol=cfgData['textCol'],
        hasHeader=cfgData['hasHeader']
    )
    
    # prepare dataloader
    loader = DataLoader(
        dataset,
        batch_size=cfgTrain['batchSize'],
        collate_fn=dataset.collate_fn,
        num_workers=cfgTrain['numWorkers']
    )
    
    # Instantiate model
    device = torch.device(cfgTrain.get('device','cpu'))
    model = T9Labeler(
        digitVocabSize=len(digit2id),
        charVocabSize=len(char2id),
        dModel=cfgModel['dModel'],
        nhead=cfgModel['nhead'],
        numLayers=cfgModel['numLayers'],
        dimFeedForward=cfgModel['dimFeedforward'],
        dropout=cfgModel['dropout'],
        maxLen=cfgModel['maxLen']
    ).to(device)
    
    # Loss
    padID = char2id.get('<pad>')
    criterion = nn.CrossEntropyLoss(ignore_index=padID)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfgTrain['lr'],
        weight_decay=cfgTrain.get('weightDecay', 1e-2)
    )
    
    # Learning rate schedule: warmup + inverse sqrt
    #total_steps = (cfgTrain['numEpochs'] * len(loader))
    warmup_steps = cfgTrain.get('warmupSteps', 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step+1)/warmup_steps, math.sqrt(warmup_steps/(step+1)))
    )
    
    #resume from checkpoint
    startBatch = 0
    ckptPath = cfgTrain['checkpointPath']
    if cfgTrain['resume'] and os.path.exists(ckptPath):
        print(f"Resuming training from {ckptPath}")
        startBatch = utils.loadCheckpoint(ckptPath, model, optimizer, scheduler)
        if not cfgTrain['continue']:
            startBatch = 0
        print(f"Training will start on batch{startBatch}")
    else:
        print("Starting training from scratch.")
    
    #Training loop
    #bestCer = float('inf')
    evalInterval = cfgTrain.get('evalInterval',5000)
    saveInterval = cfgTrain.get('saveInterval',10000)
    runningLoss = 0.0
    batchCounter=0
    charAcc, wordAcc, sentAcc = eval(model, device)
    utils.writeLogs(batchCounter, runningLoss, charAcc, wordAcc, sentAcc)
    
    model.train() #set training mode of model
    for step,batch in enumerate(tqdm(loader, desc=f"Training batch {batchCounter}")):
        batchCounter += 1
        if batchCounter < startBatch:
            continue #pass trained batch
        inputIDs = batch['inputIDs'].to(device)
        labelIDs = batch['labelIDs'].to(device)
        mask = batch['attentionMask'].to(device)
        
        optimizer.zero_grad()
        logits = model(inputIDs, attentionMask=mask)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labelIDs.view(-1)
        )
        loss.backward()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfgTrain.get('maxGradNorm', 1.0))
        optimizer.step()
        scheduler.step()
        
        runningLoss += loss.item()           
            
        # Debug: every 10 batches, print sample input and output
        if (step + 1) % 100 == 0:
            # take first example in batch
            inp = inputIDs[0].cpu().tolist()
            real_len  = mask[0].sum().item()
            pred_ids = logits[0].argmax(dim=-1).cpu()[:real_len].tolist()
            true_ids = labelIDs[0].cpu().tolist()
            # decode
            inp_str = ''.join(id2digit[id] for id in inp if id2digit[id] != '<pad>')
            pred_str = ''.join(id2char[id] for id in pred_ids if id2char[id] != '<pad>')
            true_str = ''.join(id2char[id] for id in true_ids if id2char[id] != '<pad>')
            print(f"Batch {step+1}: \nInput: {inp_str} \nPred: {pred_str} \nTrue: {true_str}")
            #clean cuda cache
            if device=='cuda':
                torch.cuda.empty_cache()
        
        if (step + 1) % evalInterval == 0:
            avgLoss = runningLoss/evalInterval
            print(f"[file {step+1} | Batch {step+1}] Avg loss:{avgLoss:.4f}")
            runningLoss = 0
            charAcc, wordAcc, sentAcc = eval(model, device)
            print(f'='*30)
            print(f"""Epoch {step+1}: \nAvg Epoch loss = {avgLoss:.4f}
            char accurate = {charAcc:.5f}
            word accurate = {wordAcc:.5f}
            sentence accute = {sentAcc:.5f}""")
            print(f'='*30+'\n')
            utils.writeLogs(batchCounter, avgLoss, charAcc, wordAcc, sentAcc)
            runningLoss = 0
            
        if (step + 1) % saveInterval == 0:
            ckptPath = os.path.join(cfgTrain.get('ckptDir'), f'batch{step+1}.pt')
            os.makedirs(os.path.dirname(ckptPath),exist_ok=True)
            utils.saveCheckpoint(model,optimizer,scheduler,batchCounter,ckptPath)

    
    ckptPath = os.path.join(cfgTrain.get('ckptDir'), 'final.pt')
    os.makedirs(os.path.dirname(ckptPath),exist_ok=True)
    utils.saveCheckpoint(model,optimizer,scheduler,batchCounter,ckptPath)
    print("Training completed.")
    
if __name__ == '__main__':
    train()