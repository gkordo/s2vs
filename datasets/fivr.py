import numpy as np
import pickle as pk

from collections import defaultdict, OrderedDict


class FIVR(object):

    def __init__(self, version='200k', audio=False):
        self.version = version
        self.audio = audio
        with open('data/fivr_audio.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'FIVR-' + version.upper()
        self.annotation = dataset['annotation']
        if not self.audio:
            self.queries = sorted(list(dataset[self.version]['queries']))
            self.tasks = {'retrieval': OrderedDict(
                {'DSVR': ['ND', 'DS'], 'CSVR': ['ND', 'DS', 'CS'], 'ISVR': ['ND', 'DS', 'CS', 'IS']}),
                          'detection': OrderedDict(
                              {'DSVD': ['ND', 'DS'], 'CSVD': ['ND', 'DS', 'CS'], 'ISVD': ['ND', 'DS', 'CS', 'IS']})}
        else:
            self.queries = sorted([q for q in dataset[self.version]['queries'] if 'DA' in self.annotation[q]])
            self.tasks = {'retrieval': {'DAVR': ['DA']}, 'detection': {'DAVD': ['DA']}}
        self.database = sorted(list(dataset[self.version]['database']))
        self.easy_duplicates = set(np.loadtxt('data/fivr_easy_duplicates.txt', dtype=str).tolist())

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def calculate_metric(self, y_true, y_score, gt_len):
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)

    def calculate_mAP(self, query, targets, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db) - {query}

        if len(query_gt) == 0:
            return None

        y_true, y_score = [], []
        for target, sim in targets.items():
            if target != query and target in all_db:
                y_true.append(int(target in query_gt))
                y_score.append(float(sim))

        return self.calculate_metric(y_true, y_score, len(query_gt))

    def calculate_uAP(self, similarities, all_db, relevant_labels):
        y_true, y_score, gt_len = [], [], 0
        for query, targets in similarities.items():
            if query in self.queries:
                gt_sets = self.annotation[query]
                query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
                query_gt = query_gt.intersection(all_db) - {query}

                if isinstance(targets, (np.ndarray, np.generic)):
                    targets = {v: s for v, s in zip(self.database, targets) if v in all_db}
                gt_len += len(query_gt)
                for target, sim in targets.items():
                    if target != query and target in all_db:
                        y_true.append(int(target in query_gt))
                        y_score.append(float(sim))
        return self.calculate_metric(y_true, y_score, gt_len)

    def print_results(self, results, similarities, db, hard=True):
        print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))

        print('Queries: {} videos'.format(len(similarities.keys())))
        print('Database: {} videos'.format(len(db)))
        print('-' * 5, 'normal', '-' * 5)
        for task in self.tasks['retrieval']:
            print('{} mAP: {:.4f}'.format(task, results[task]))
        print()
        for task in self.tasks['detection']:
            print('{} uAP: {:.4f}'.format(task, results[task]))
        if hard:
            print()
            print('-' * 5, 'hard', '-' * 5)
            for task in self.tasks['retrieval']:
                print('{} mAP: {:.4f}'.format(task, results[task + '_h']))
            print()
            for task in self.tasks['detection']:
                print('{} uAP: {:.4f}'.format(task, results[task + '_h']))

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set(self.database)
        else:
            all_db = set(self.database).intersection(all_db)

        results = defaultdict(list)
        for db, ext in [[all_db, ''], [all_db - self.easy_duplicates, '_h']]:
            for query, targets in similarities.items():
                if query in self.queries:
                    if isinstance(targets, (np.ndarray, np.generic)):
                        assert len(self.database) == len(targets), 'Similarities must me the same size as the dataset'
                        targets = {v: s for v, s in zip(self.database, targets) if v in db}
                    for task, labels in self.tasks['retrieval'].items():
                        mAP = self.calculate_mAP(query, targets, db, relevant_labels=labels)
                        if mAP is not None: results[task + ext].append(mAP)
            for task in self.tasks['retrieval']:
                results[task + ext] = np.mean(results[task + ext])

            for task, labels in self.tasks['detection'].items():
                results[task + ext] = self.calculate_uAP(similarities, db, relevant_labels=labels)
        if verbose:
            self.print_results(results, similarities, all_db, hard=not self.audio)
        return results
