import subprocess
import sys
import matplotlib.pyplot as plt

def comb(n, k):
    assert(n >= 0 and k >= 0)

    if k == 0 or k == n:
        return 1
    elif n < k:
        return 0
    elif mem_comb[n][k] != -1:
        return mem_comb[n][k]
    else:
        mem_comb[n][k] = comb(n-1, k-1) + comb(n-1, k)
        return mem_comb[n][k]


num_var = 30
delta = 0.2
eps = 0.8
filename = "tempcard.opb"
mem_comb = [[-1 for _ in range(num_var+1)] for _ in range(num_var+1)]

header = f"* #variable= {num_var}\n\n" 
# generate lhs
lhs = ""
benchmark = []
cnts = []
up = []
low = []
for var in range(num_var):
    lhs += f"1 x{var+1} "

for k in range(num_var+2):
    # generate opb file
    with open(filename, "w") as f:
        f.write( f"{header}{lhs}>= {k};\n")
    # print(f"{header}{lhs} >= {k};")
    
    # count
    call_args = ['approxmc', 
                '-v', '0', 
                '--start', '0', 
                '--seed', '1',
                '--epsilon', str(eps),
                '--delta', str(delta)
                # '--log', str(args.log)
                ]
    call_args += [filename]
    
    # usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)

    # pipe_stdout = None if args.verbose else subprocess.PIPE
    # print(' '.join(call_args))
    pipe_stdout = subprocess.PIPE
    process = subprocess.Popen(call_args, stdout=pipe_stdout)
    try:
        stdout = process.communicate(timeout=1000)[0]
    except subprocess.TimeoutExpired:
        process.kill()
        stdout = b'-1\n'
        print("Counting Timeout !")

    # usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    # cpu_time = usage_end.ru_utime - usage_start.ru_utime

    if stdout:
        cnt = stdout.decode(sys.stdout.encoding).split('\n')[-2].split(' ')[-1]
        cnts.append(int(cnt))
        benchmark.append(benchmark[-1] - comb(num_var, k-1) if k else 2**num_var)
        low.append(int(benchmark[k]/(1+eps)))
        up.append(int(benchmark[k]*(1+eps)))

        print(f'{low[k]} <= {cnts[k]} ({benchmark[k]}) <= {up[k]}' )
        
# reverse, from small to large
low.reverse()
benchmark.reverse()
up.reverse()
cnts.reverse()

# plot
x = list(range(-1, num_var+1))
plt.plot(x, low, 'g', x, benchmark, 'r', x, up, 'b', x, cnts, 'k')
plt.ylabel('counts')
plt.xlabel('k')
plt.show()
