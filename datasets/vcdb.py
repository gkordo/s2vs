import numpy as np
import pickle as pk

from collections import defaultdict


class VCDB(object):

    def __init__(self, distractors=False):
        self.name = 'VCDB'
        if distractors:
            self.name += '-DIST'
        with open('data/vcdb.pickle', 'rb') as f:
            dataset = pk.load(f) 
        self.queries = dataset['queries']
        self.positives = dataset['positives']
        self.dataset = dataset['dataset']
        self.distractors = distractors

    def get_queries(self):
        return self.queries

    def get_database(self):
        if not self.distractors:
            return self.queries
        return self.queries + self.dataset

    def calculate_metric(self, y_true, y_score, gt_len):
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)

    def calculate_mAP(self, query, targets, all_db):
        query_gt = self.positives[query].intersection(all_db)

        y_true, y_score = [], []
        for target, sim in targets.items():
            if target != query and target in all_db:
                y_true.append(int(target in query_gt))
                y_score.append(float(sim))

        return self.calculate_metric(y_true, y_score, len(query_gt))

    def calculate_uAP(self, similarities, all_db):
        y_true, y_score, gt_len = [], [], 0
        for query in self.queries:
            if query in similarities:
                query_gt = self.positives[query].intersection(all_db)
                gt_len += len(query_gt)
                for target, sim in similarities[query].items():
                    if target != query and target in all_db:
                        y_true.append(int(target in query_gt))
                        y_score.append(float(sim))

        return self.calculate_metric(y_true, y_score, gt_len)

    def evaluate(self, similarities, all_db=None, verbose=True):
        mAP, not_found = [], 0
        if all_db is None:
            all_db = set(self.get_database())

        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                mAP += [self.calculate_mAP(query, similarities[query], all_db)]

        uAP = self.calculate_uAP(similarities, all_db)

        if verbose:
            print('=' * 5, 'VCDB Dataset', '=' * 5)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(mAP)))
            print('uAP: {:.4f}'.format(uAP))
        return {'VCDB_mAP': np.mean(mAP), 'VCDB_uAP': uAP}
