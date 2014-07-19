__author__ = 'Aubrey'

import numpy as np
import subprocess
from HelperFunctions import make_dir_for_file_name
from HelperFunctions import check_input


class RankLibConfigs(object):
    def __init__(self):
        pass


def save_in_letor_format(X, Y, target_ids, file_name):
    check_input(X, Y, target_ids)
    num_items = len(target_ids)
    make_dir_for_file_name(file_name)
    with open(file_name, 'w') as f:
        for index in range(num_items):
            xi = X[index,:]
            yi = Y[index]
            target_id = target_ids[index]
            score = yi
            s = str(score) + ' ' + 'qid:' + str(target_id)
            for feat_index in range(len(xi)):
                s += ' ' + str(feat_index) + ':' + str(xi[feat_index])
            s += '\n'
            f.write(s)

def load_letor_scores(file_name, num_instances):
    scores = np.ndarray(num_instances)
    with open(file_name, 'r') as f:
        index = 0
        for s in f:
            vals = s.split()
            score = float(vals[-1])
            scores[index] = score
            index += 1
    return scores

def run_ranking(configs):
    k = configs.k
    num_folds = 5
    ranker_opt = configs.ranker_opt
    metric_opt = 'P@' + str(k)
    score_max = 1
    args = [
        'java',
        '-jar',
        'RankLib/RankLib-2.1-patched.jar',
        '-ranker',
        str(ranker_opt),
        '-metric2t',
        str(metric_opt),
        '-gmax',
        str(score_max),
        '-tree',
        '500',
        '-estop',
        '50'
    ]
    if hasattr(configs, 'model_file_name'):
        args.append('-load')
        args.append(configs.model_file_name)
    if hasattr(configs, 'save_file_name'):
        args.append('-save')
        args.append(configs.save_file_name)
    if hasattr(configs, 'train_file_name'):
        args.append('-train')
        args.append(configs.train_file_name)
        if hasattr(configs, 'cv_file_name'):
            args.append('-validate')
            args.append(configs.cv_file_name)
        # I'm not sure if this is valuable/necessary
        # else:
        #     args.append('-kcv')
        #     args.append(str(num_folds))
    if hasattr(configs, 'test_file_name'):
        args.append('-rank')
        args.append(configs.test_file_name)
        args.append('-score')
        args.append(configs.score_file_name)
    ret_val = subprocess.call(args)
    assert ret_val == 0
    pass