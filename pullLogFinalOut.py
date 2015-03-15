import os
#os.getcwd()
root = '/Volumes'
f = open(r'LogFinalOut.txt','w')
f.write('file\tnumInput\tuniquelyMapped\tuniquePercent\n')
index = [5,8,9]
for subdir, dirs, files in os.walk(root):
    for file in files:
        if file != 'Log.final.out':
            continue
        filename = os.path.join(subdir,file)
        txt = open(filename).read().split('\n')
        stat = '\t'.join([txt[i].strip().split('\t')[1] for i in index])
        f.write(filename + '\t' + stat + '\n')
f.close()
