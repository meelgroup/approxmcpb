import numpy as np
from pysat.card import *
from pysat.formula import CNFPlus
n = 5
opb_file = f"latin_{n}.opb"
cnf_file = f"latin_{n}.cnf"

opb = ""
out_cnf = CNFPlus() 

var = np.zeros((n,n,n), dtype=np.int32)
cnt= 1
for i in range(n):
    for j in range(n):
        for k in range(n):
            var[i, j, k] = cnt
            cnt += 1

top_id = n**3
for i in range(n):
    for j in range(n):
        lits = []
        for k in range(n):
            opb += " +1 x"+str(var[i, j, k])
            lits.append(int(var[i, j, k]))
        opb += " = 1 ;\n"
        cnf = CardEnc.atmost(lits=lits, bound=1, top_id=top_id, encoding=EncType.cardnetwrk)
        top_id = max(cnf.nv, top_id)
        out_cnf.extend(cnf)

for i in range(n):
    for k in range(n):
        lits = []
        for j in range(n):
            opb += " +1 x"+str(var[i, j, k])
            lits.append(int(var[i, j, k]))
        opb += " = 1 ;\n"
        cnf = CardEnc.atleast(lits=lits, bound=1, top_id=top_id, encoding=EncType.cardnetwrk)
        top_id = max(cnf.nv, top_id)
        out_cnf.extend(cnf)

for j in range(n):
    for k in range(n):
        lits = []
        for i in range(n):
            opb += " +1 x"+str(var[i, j, k])
            lits.append(int(var[i, j, k]))
        opb += " = 1 ;\n"
        cnf = CardEnc.atleast(lits=lits, bound=1, top_id=top_id, encoding=EncType.cardnetwrk)
        top_id = max(cnf.nv, top_id)
        out_cnf.extend(cnf)

# opb encoding
header = "* #variable= "+str(n**3)+" #constraint= "+ str(3*n**2)+"\n"
with open(opb_file, 'w') as f:
    f.write(header+opb)

# cnf encoding
header = f"p cnf {n**3} {3*n**2}\n"

# sampling set
comment = []
for i in range(n**3//10):
    cm = ['c ind']
    for j in range(10):
        cm.append(str(i*10 + j + 1))
    cm.append('0')

    comment.append(' '.join(cm))

cm = ['c ind']
for i in range(n**3//10*10, n**3):
    cm.append(str(i + 1))
cm.append('0')

comment.append(' '.join(cm))

with open(cnf_file, 'w') as fp:
    out_cnf.to_fp(fp, comment)
