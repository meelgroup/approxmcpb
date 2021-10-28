# input: path to dimacs file (cnf)
# output: opb file (pb) in the same directory as input
# 
# eg: python cnf2pb.py ./test.dimacs    (will create ./test.opb)

import sys

assert(len(sys.argv) > 1)
cnf_file = sys.argv[1]
opb_file = '.'.join(cnf_file.split('.')[:-1] + ['opb'])
print('input: ', cnf_file)
print('output:', opb_file)

num_var, num_cls = 0, 0
out_lines = []

# parse dimacs file
with open(cnf_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        if line[0] == 'p':
            assert(line[1] == 'cnf')
            num_var, num_cls = int(line[2]), int(line[3])
        elif line[0] == 'c':
            line[0] = '*'
            out_lines += [' '.join(line)]
        else:
            line = [f'1 ~x{v[1:]}' if v[0] == '-' else f'1 x{v}' for v in line[:-1]]
            line += ['>= 1 ;\n']
            out_lines += [' '.join(line)]

# write opb file
with open(opb_file, 'w') as f:
    f.write(f'* #variable= {num_var} #constraint= {num_cls}\n')
    f.writelines(out_lines)

