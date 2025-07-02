import os
import glob
import json
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
import csv

class T9Dataset(IterableDataset):
    def __init__(
        self,
        dataDir: str,
        digitVocabPath: str,
        charVocabPath: str,
        sentenceCheck: bool = False,
        maxSentenceLen: int = 20,
        maxWordLen: int = 20,
        csvDelimiter: str = ',',
        digitCol: int = 0,
        textCol: int = 1,
        hasHeader: bool = True
    ):
        # Collect CSV file paths in sorted order
        pattern = os.path.join(dataDir, '*.csv')
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise ValueError(f"No CSV files found in directory {dataDir}")
        
        # Load vocabularies
        with open(digitVocabPath, 'r', encoding='utf-8') as f:
            self.digit2id = json.load(f)
        with open(charVocabPath, 'r', encoding='utf-8') as f:
            self.char2id = json.load(f)

        # Special token IDs
        self.padInputId = self.digit2id.get('<pad>')
        self.padLabelId = self.char2id.get('<pad>')

        # Configure
        self.maxSentenceLen = maxSentenceLen
        self.maxWordLen = maxWordLen
        self.delimiter = csvDelimiter
        self.digitCol = digitCol
        self.textCol = textCol
        self.hasHeader = hasHeader
        self.sentenceCheck = sentenceCheck
    
    def parseRow(self, row):
        try:
            digits = row[self.digitCol].strip()
            text = row[self.textCol].strip()
        except IndexError:
            return None
        
        if self.sentenceCheck:
            #Ensure equal length
            if len(digits) != len(text):
                return None
            sentence = text.split()
            #Filter by length
            if self.maxSentenceLen and len(sentence) > self.maxSentenceLen: #限制句子的最大单词量
                return None
            if self.maxWordLen and max([len(word) for word in sentence]) > self.maxWordLen:  #限制单词的的最大长度
                return None
            
        #Convert to IDs
        inputIds = torch.tensor(
            [self.digit2id.get(ch, self.padInputId) for ch in digits],
            dtype=torch.long
        )
        labelIds = torch.tensor(
            [self.char2id.get(ch, self.padLabelId) for ch in text]
        )
        return inputIds, labelIds
    
    def __iter__(self):
        #check the worker informeation
        workerInfo = get_worker_info()
        if workerInfo is None: #only 1 worker
            filesToProcess = self.files
        else:
            num_workers = workerInfo.num_workers
            worker_id = workerInfo.id
            # 拆分文件给不同 worker，防止重复读
            files_to_process = self.files[worker_id::num_workers]
            
        for filePath in files_to_process:
            with open(filePath, 'r',encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                if self.hasHeader: #pass the header
                    next(reader)
                for row in reader:
                    parsed = self.parseRow(row)
                    if parsed: #check if the row valid
                        inputIDs, labelIDs = parsed
                        yield {'inputIDs': inputIDs, 'labelIDs': labelIDs}
                        

    def collate_fn(self, batch):
        #padding for batch
        #determine bach max length
        batchSize = len(batch)
        maxLen = max(item['inputIDs'].size(0) for item in batch)
        #prepare containers
        inputIDs = torch.full((batchSize, maxLen), self.padInputId, dtype=torch.long)
        labelIDs = torch.full((batchSize, maxLen), self.padLabelId, dtype=torch.long)
        attentionMask = torch.zeros((batchSize, maxLen), dtype=torch.bool)
        #put data into containers
        for i, item in enumerate(batch):
            sLen = item['inputIDs'].size(0)
            inputIDs[i, :sLen] = item['inputIDs']
            labelIDs[i, :sLen] = item['labelIDs']
            attentionMask[i, :sLen] = 1
        #return containers
        return {
            'inputIDs': inputIDs,
            'labelIDs': labelIDs,
            'attentionMask': attentionMask
        }