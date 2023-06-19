import numpy as np
import pickle as pk

class EVVE(object):

    def __init__(self):
        with open('data/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'EVVE'
        self.events = dataset['annotation']
        self.queries = sorted(list(dataset['queries']))
        self.database = sorted(list(dataset['database']))
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def calculate_metric(self, y_true, y_score, gt_len):
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)

    def calculate_mAP(self, similarities, all_db):
        results = {e: [] for e in self.events}
        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                targets = similarities[query]
                if isinstance(targets, (np.ndarray, np.generic)):
                    targets = {v: s for v, s in zip(self.database, targets) if v in all_db}
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(targets.keys(), key=lambda x: targets[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)
        return results, not_found
    
    def calculate_uAP(self, similarities, all_db):
        y_true, y_score, gt_len = [], [], 0
        for query in self.queries:
            if query in similarities:
                targets = similarities[query]
                if isinstance(targets, (np.ndarray, np.generic)):
                    targets = {v: s for v, s in zip(self.database, targets) if v in all_db}
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                gt_len += len(pos)
                
                for target, sim in targets.items():
                    if target in all_db:
                        y_true.append(target in pos)
                        y_score.append(sim)

        return self.calculate_metric(y_true, y_score, gt_len)
    
    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        results, not_found = self.calculate_mAP(similarities, all_db)
        uAP = self.calculate_uAP(similarities, all_db)
        if verbose:
            print('=' * 18, 'EVVE Dataset', '=' * 18)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
            print('-' * 50)
        mAP = []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            mAP.extend(results[evname])
            if verbose:
                print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(np.sum(results[evname]) / nq))

        if verbose:
            print('=' * 50)
            print('overall mAP = {:.4f}'.format(np.mean(mAP)))
            print('overall uAP = {:.4f}'.format(uAP))
        return {'EVVE_mAP': np.mean(mAP), 'EVVE_uAP': uAP}
