import re
import csv
import os
import glob
import utils
cfg = utils.loadConfig()
cfgData = cfg['data']





dataDir = cfgData['sourceDataDir']
trainDataDir = cfgData['trainDataDir']
pattern = os.path.join(dataDir, '*')
files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
files.sort()
for i,file in enumerate(files):
    # 打开输出 CSV，写入标题行
    outputDir = os.path.join(trainDataDir, f'data{i:02d}.csv')
    with open(outputDir, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['input', 'output'])

        with open(file, 'r', encoding='utf-8') as f:
            counter = 0
            for line in f:
                doc = line.split(',')
                for sent in doc:
                    #数字を含む文を処理しない
                    if re.search(r'\d', sent): continue
                    
                    #去掉句尾所有的句号问号感叹号
                    sent = re.sub(r'[.?!]+$', '', sent.strip())
                    
                    #不允许出现字母 空格 逗号 句号以外的符号
                    if re.search(r"[^a-zA-Z\s,.]", sent): continue
                    
                    #删除逗号句号
                    sent = re.sub(r'[^\w\s]', '', sent).lower().strip()
                    
                    #去除空格
                    sent = re.sub(r'\s+', ' ', sent).strip()
                    
                    #transform to number by T9-rule
                    sT9 = utils.textToNumber(sent)
                    if not sT9:
                        continue
                    
                    #filter by length
                    splitSent = sent.split()
                    length = len(splitSent)
                    if length>cfgData['maxSentenceLen'] or length<cfgData['minSentenceLen']:
                        continue
                    if max(len(word) for word in splitSent)>cfgData['maxWordLen']:
                        continue
                    
                    writer.writerow([sT9, sent])
                counter += 1
                if counter % 10000 == 0:
                    print(f"{i}-th file: {counter} sentences have been processed...")
