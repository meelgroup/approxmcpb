from pysat.card import *
from pysat.formula import CNFPlus
import argparse
import random
import resource
import subprocess
import time
import sys
import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
from multiprocessing import Pool, Process, Queue, Manager, cpu_count
import copy
import os

timeout = 1000
time_penalty = 10
eps = 1e-10
pp = pprint.PrettyPrinter(indent=8)

def create_parser():
    parser = argparse.ArgumentParser(description='compare performance')
    parser.add_argument('--path-cnf', type=str, default='./temp.cnf', help='path of cnf file')
    parser.add_argument('--path-opb', type=str, default='./temp.opb', help='path of opb file')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--max-var', type=int, default=10, help='maximum number of variables')
    parser.add_argument('--num-constr', type=int, default=3, help='number of constraints')
    parser.add_argument('--cls-width', type=int, default=3, help='number of literals in a clause')
    parser.set_defaults(func=run)

    subparsers = parser.add_subparsers(dest='subparser_name')
    # train bnn
    mul_parser = subparsers.add_parser('mult-run', help='multiple runs')
    mul_parser.add_argument('--range-var', type=str, default='5,10', help='range for max-var')
    mul_parser.add_argument('--step-var', type=int, default=1, help='increasing step in the range for max-var')
    mul_parser.add_argument('--range-clause', type=str, default='5,10', help='range for cls-width')
    mul_parser.add_argument('--step-clause', type=int, default=1, help='increasing step in the range for cls-width')
    mul_parser.add_argument('--range-constr', type=str, default='5,10', help='range for num-constr')
    mul_parser.add_argument('--step-constr', type=int, default=1, help='increasing step in the range for num-constr')
    mul_parser.add_argument('--cnt-cnf', type=str, default='approxmc', help='the path to the CNF counter')
    mul_parser.add_argument('--cnt-pb', type=str, default='approxmc', help='the path to the PB counter')
    mul_parser.add_argument('--epsilon', type=float, default=0.8, metavar='EPS',
                            help='epsilon of PAC argument calling approxmc')
    mul_parser.add_argument('--delta', type=float, default=0.2, metavar='DELTA',
                            help='delta of PAC argument calling approxmc')
    mul_parser.add_argument('--multiprocess', type=int, default=0, help='number of threads. If > 0, run by multiprocessing. Bounded by cpu_count().')
    mul_parser.set_defaults(func=mult_run)

    return parser


def encode_constr(coefs, lits, deg):
    # by default cmp is >=
    assert(len(coefs) == len(lits))

    line = []
    for coef, lit in zip(coefs, lits):
        if coef != 0:
            line += [str(coef), ('x%d' % lit) if lit>0 else ('~x%d' % -lit)]
    
    if len(line) == 0:
        return ""

    line += ['>=', str(deg), ';\n']    
    line = ' '.join(line)
    return line


def run(args):
    out_cnf = CNFPlus() 
    top_id = args.max_var
    opb_lines = []

    for _ in range(args.num_constr):
        lits = []
        for _ in range(args.cls_width):
            lits.append(random.randrange(args.max_var) + 1)
        bound = random.randrange(args.cls_width+1)
        # print(lits)
        # print(bound)

        # cnf
        cnf = CardEnc.atleast(lits=lits, bound=bound, top_id=top_id, encoding=EncType.cardnetwrk)
        # print(cnf.nv)
        top_id = max(cnf.nv, top_id)
        # print(top_id)
        # print(cnf.clauses)

        out_cnf.extend(cnf)

        # opb
        coefs = [1] * args.cls_width
        opb_lines.append(encode_constr(coefs, lits, bound))
    # resolve the corner case when max_var is not used, resulting in num_var < maximum ind var
    out_cnf.nv = max(out_cnf.nv, args.max_var)

    # sampling set
    comment = []
    opb_comment = []
    for i in range(args.max_var//10):
        cm = ['c ind']
        for j in range(10):
            cm.append(str(i*10 + j + 1))
        cm.append('0')

        comment.append(' '.join(cm))
        opb_comment.append(' '.join(['* ind'] + cm[1:]) + '\n')

    cm = ['c ind']
    for i in range(args.max_var//10*10, args.max_var):
        cm.append(str(i + 1))
    cm.append('0')

    comment.append(' '.join(cm))
    opb_comment.append(' '.join(['* ind'] + cm[1:]) + '\n')

    # write cnf file
    # print(comment)
    # print(out_cnf.clauses)
    path_cnf = args.path_cnf if not args.multiprocess else f'tmp/{args.num_constr}-{args.cls_width}.cnf'
    with open(path_cnf, 'w') as fp:
        out_cnf.to_fp(fp, comment)

    # write opb file
    path_opb = args.path_opb if not args.multiprocess else f'tmp/{args.num_constr}-{args.cls_width}.opb'
    with open(path_opb, 'w') as fp:
        fp.write(f'* #variable= {args.max_var} #constraint= {args.num_constr}\n*\n')
        fp.write(f'{"".join(opb_comment)}*\n')
        fp.write(''.join(opb_lines))


def mult_run(args):
    # parse arguments
    range_var = [int(x) for x in args.range_var.split(',')]
    assert(len(range_var) == 2)
    var_st, var_ed = range_var

    range_clause = [int(x) for x in args.range_clause.split(',')]
    assert(len(range_clause) == 2)
    clause_st, clause_ed = range_clause

    range_constr = [int(x) for x in args.range_constr.split(',')]
    assert(len(range_constr) == 2)
    constr_st, constr_ed = range_constr

    assert(args.multiprocess >= 0)
    if args.multiprocess:
        # procs = []
        # queue = Queue()
        manager = Manager()
        queue = manager.Queue()
        pool = Pool(processes=min(args.multiprocess, cpu_count()))
    rec_cnf_t, rec_cnf_cnt, rec_pb_t, rec_pb_cnt = [], [], [], []
    for num_constr in range(constr_st, constr_ed+1, args.step_constr):
        print(num_constr)
        args.num_constr = num_constr
        cnf_t, cnf_cnt, pb_t, pb_cnt = [], [], [], []
        # args.max_var = var
        for cls in range(clause_st, clause_ed+1, args.step_clause):
            print(cls, end=' ')
            args.cls_width = cls
            run(args)

            for idx in range(2):
                args.type_idx = idx

                if args.multiprocess:
                    # proc = Process(target=execute, args=(args, queue))
                    # procs.append(proc)
                    # proc.start()
                    pool.apply_async(execute, args=(copy.copy(args), queue))
                else:
                    cnts, cpu_time = execute(args)

                    if idx == 0:
                        cnf_t.append(cpu_time)
                        cnf_cnt.append(cnts)
                    else:
                        pb_t.append(cpu_time)
                        pb_cnt.append(cnts)

                    # record output info
                    # with open(args.log_path, 'ab') as f:
                    #     f.write(stdout)

                    # if args.multiprocess:
                    #     proc_time = time.process_time() - time_st + cpu_time
                    #     qout.put([args.cnt_layer, cnts, proc_time])
            
                    # return cnts
            # print(cnf_cnt)
            # print(pb_cnt)
            if not args.multiprocess:
                assert(cnf_cnt[-1] == pb_cnt[-1] or cnf_cnt[-1] == '-1' or pb_cnt[-1] == '-1')

        print()
        assert(len(cnf_t) == len(cnf_cnt))
        assert(len(pb_t) == len(pb_cnt))
        assert(len(cnf_t) == len(pb_t))
        rec_cnf_t.append(cnf_t)
        rec_cnf_cnt.append(cnf_cnt)
        rec_pb_t.append(pb_t)
        rec_pb_cnt.append(pb_cnt)

    # postprocess when multiprocess
    if args.multiprocess:
        pool.close()
        pool.join()

        rec_cnf_t = [[None for _ in range(clause_st, clause_ed+1, args.step_clause)] for _ in range(constr_st, constr_ed+1, args.step_constr)]
        rec_pb_t = [[None for _ in range(clause_st, clause_ed+1, args.step_clause)] for _ in range(constr_st, constr_ed+1, args.step_constr)]
        rec_cnf_cnt = [[None for _ in range(clause_st, clause_ed+1, args.step_clause)] for _ in range(constr_st, constr_ed+1, args.step_constr)]
        rec_pb_cnt = [[None for _ in range(clause_st, clause_ed+1, args.step_clause)] for _ in range(constr_st, constr_ed+1, args.step_constr)]
        base_constr = len(rec_cnf_t)
        base_cls = len(rec_cnf_t[0])
        for _ in range(base_constr*base_cls*2):
            id_constr, id_cls, id_type, cnts, cpu_time = queue.get()
            id_constr = (id_constr - constr_st) // args.step_constr
            id_cls = (id_cls - clause_st) // args.step_clause
            if id_type == 0:
                rec_cnf_t[id_constr][id_cls] = cpu_time
                rec_cnf_cnt[id_constr][id_cls] = cnts
            else:
                rec_pb_t[id_constr][id_cls] = cpu_time
                rec_pb_cnt[id_constr][id_cls] = cnts
            # procs[id_constr*base_constr + id_cls*base_cls + id_type].join()
        
        # assert counts match
        for i in range(math.floor((constr_ed+1-constr_st) / args.step_constr)):
            for j in range(math.floor((clause_ed+1-clause_st) / args.step_clause)):
                assert(rec_cnf_cnt[i][j] == rec_pb_cnt[i][j] or rec_cnf_cnt[i][j] == '-1' or rec_pb_cnt[i][j] == '-1')
                
    ''' 
    print('## time ##')
    for rec in rec_cnf_t:
        print(rec)
    print('##########')
    for rec in rec_pb_t:
        print(rec)
    '''
    # compute acceleration rate
    acc = []
    for i in range(len(rec_cnf_t)):
        acc.append([])
        for j in range(len(rec_cnf_t[i])):
            if rec_cnf_t[i][j] == rec_pb_t[i][j] == 0:
                acc[-1].append(1)
            else:
                acc[-1].append(rec_cnf_t[i][j] / (rec_pb_t[i][j] + eps))

    # print(acc)
    # print('## count ##')
    # print(rec_cnf_cnt)
    # print(rec_pb_cnt)
    
    # save data
    filename = f'max-var-{args.max_var}_cls-{clause_st}-{args.step_clause}-{clause_ed}_constr-{constr_st}-{args.step_constr}-{constr_ed}'
    xlabels = list(range(constr_st, constr_ed+1, args.step_constr))
    ylabels = list(range(clause_st, clause_ed+1, args.step_clause))
    pd.DataFrame(np.array(acc), index=xlabels).to_csv(f'data/acc/{filename}.csv', header=ylabels, index_label='num of constraints\\width of clause', float_format='%.3e')
    pd.DataFrame(np.array(rec_cnf_cnt), index=xlabels).to_csv(f'data/cnt/{filename}-cnf.csv', header=ylabels, index_label='num of constraints\\width of clause', float_format='%.3e')
    pd.DataFrame(np.array(rec_pb_cnt), index=xlabels).to_csv(f'data/cnt/{filename}-pb.csv', header=ylabels, index_label='num of constraints\\width of clause', float_format='%.3e')
    pd.DataFrame(np.array(rec_cnf_t), index=xlabels).to_csv(f'data/time/{filename}-cnf.csv', header=ylabels, index_label='num of constraints\\width of clause', float_format='%10.3f')
    pd.DataFrame(np.array(rec_pb_t), index=xlabels).to_csv(f'data/time/{filename}-pb.csv', header=ylabels, index_label='num of constraints\\width of clause', float_format='%10.3f')

    # preprocess
    cbarlabels = ['log acceleration rate', 'cnf log_time', 'pb log_time', 'cnf log(number of solutions + 1)', 'pb log(number of solutions + 1)']
    rec_cnf_cnt = [[float(x)+1 for x in cnts] for cnts in rec_cnf_cnt]
    rec_pb_cnt = [[float(x)+1 for x in cnts] for cnts in rec_pb_cnt]
    z = [acc, rec_cnf_t, rec_pb_t, rec_cnf_cnt, rec_pb_cnt]
    assert(len(z) == len(cbarlabels))
    fig, axs = plt.subplots(1, len(z))
    # plot the heat map
    for idx in range(len(z)):
        z_value = np.log(np.array(z[idx]) + eps)
        # fig, ax = plt.subplots()
        ax = axs[idx]
        # im = ax.imshow(acc)
        ax.set_xlabel('width of clause')
        ax.set_ylabel('number of constraints')
        # xlabels = list(range(var_st, var_ed, args.step_var))
        xlabels = list(range(constr_st, constr_ed+1, args.step_constr))
        ylabels = list(range(clause_st, clause_ed+1, args.step_clause))
        # xlabels = []
        # ylabels = []
        im, cbar = heatmap(z_value, xlabels, ylabels, ax, cbarlabel=cbarlabels[idx])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # texts = annotate_heatmap(im, valfmt="{x:.2f}")
        # fig.tight_layout()
    fig.subplots_adjust(wspace=1)

    pickle.dump(fig, open(f'data/figs/{filename}.pickle', 'wb'))
    plt.show()


def execute(args, qout=None):
    call_args = [args.cnt_cnf if args.type_idx == 0 else args.cnt_pb, 
                # '-v', str(args.verbose), 
                '--start', '0', 
                '--seed', str(args.seed),
                # '--th', str(args.threads),
                '--epsilon', str(args.epsilon),
                '--delta', str(args.delta),
                # '--sparse', str(int(args.sparse_xor))
                ]
    if args.type_idx == 0:
        path = args.path_cnf if not args.multiprocess else f'tmp/{args.num_constr}-{args.cls_width}.cnf'
    else:
        path = args.path_opb if not args.multiprocess else f'tmp/{args.num_constr}-{args.cls_width}.opb'
    call_args += [path]

    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)

    # pipe_stdout = None if args.verbose else subprocess.PIPE
    pipe_stdout = subprocess.PIPE
    process = subprocess.Popen(call_args, stdout=pipe_stdout)
    try:
        stdout = process.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        process.kill()
        stdout = b'-1\n'
        print("Counting Timeout !")
        cpu_time = timeout * time_penalty
    else:
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_time = usage_end.ru_utime - usage_start.ru_utime

    cnts = stdout.decode(sys.stdout.encoding).split('\n')[-2].split(' ')[-1]

    if args.multiprocess:
        qout.put([args.num_constr, args.cls_width, args.type_idx, cnts, cpu_time])
        os.remove(f'tmp/{args.num_constr}-{args.cls_width}.{"cnf" if args.type_idx == 0 else "opb"}')

    return cnts, cpu_time

    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def __main__():
    parser = create_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    args.func(args)

__main__()
