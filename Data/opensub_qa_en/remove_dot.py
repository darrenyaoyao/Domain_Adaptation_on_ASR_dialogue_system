import sys
import re

fin = sys.argv[1]
fout = sys.argv[2]
data = [ ]
with open(fin,'r') as infile:
    for line in infile:
        regex = re.compile('\.(?!\d)')
        t = regex.sub('',line)
        t = t.strip()
        t = regex.sub('',t)
        t.replace('  ', ' ')
        t = t.strip()
        data.append(t)

with open(fout,'w') as outfile:
    for line in data:
        outfile.write(line+'\n')
