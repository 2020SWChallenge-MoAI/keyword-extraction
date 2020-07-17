import os
import re


input_dir = os.getcwd() + '/data/raw'
output_dir = os.getcwd() + '/data/processed'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for filename in os.listdir(input_dir):
    title = ''
    str = ''
    with open(os.path.join(input_dir, filename)) as f:
        title = f.readline().strip()

        # join all lines
        str = ''.join(line.strip() for line in f)

        # remove ending
        str = re.sub('[〈<⟪].*[〉>⟫]$', '', str)
        str = re.sub('-.*\.$', '', str)
        
        # regularize quotation mark
        str = re.sub('[‘’]', "'", str)
        str = re.sub('[“”]', '"', str)

        # append period
        if not str.endswith('.'):
            str += '.'
            
    with open(os.path.join(output_dir, filename), mode = 'w') as f:
        f.write(f'@title\n{title}\n')
        f.write(f'@content\n{str}')
