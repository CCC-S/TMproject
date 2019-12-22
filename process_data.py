import os
import re

def read_data_by_files(path):
    filelist = os.listdir(path)
    filelist.sort()
    for filename in filelist:
        temp = []
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            file_text = f.read()
            obj = re.compile('-\d\W\d*')
            epoch = re.compile('\d+\/100')
            tempE = epoch.findall(file_text)
            temp = obj.findall(file_text)
            store_data([temp, tempE], filename, path)



def store_data(data, name, path):
    out_file = "acc-" + str(name) + ".txt"
    with open(os.path.join(path, out_file), 'w', encoding='utf-8') as fout:
        line, lineE = data[:]
        for i in range(len(line)):
            e = lineE[i].split('/')[0]
            acc = line[i][1:]
            fout.write('%s, %s'%(e, acc))
            fout.write('\n')

read_data_by_files('./log')
#read_data_by_files('../result/hyperopt/log_lr/')
#read_data_by_files('../result/hyperopt/log_nb/')
#read_data_by_files('../result/hyperopt/log_sgd/')