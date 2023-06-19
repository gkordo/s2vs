import numpy as np
import pickle as pk


class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('data/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'CC_WEB_VIDEO'
        self.index = dataset['index']
        self.queries = list(dataset['queries'])
        self.database = sorted(list(map(str, self.index.keys())))
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def calculate_mAP(self, similarities, all_db, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                if isinstance(res, (np.ndarray, np.generic)):
                    res = {v: s for v, s in zip(self.database, res) if v in all_db}
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    if video_id not in self.index: continue
                    video = self.index[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set(self.database)

        if verbose:
            print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
            not_found = len(set(self.queries) - similarities.keys())
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos'.format(len(all_db)))

        mAP = self.calculate_mAP(similarities, all_db, all_videos=False, clean=False)
        mAP_star = self.calculate_mAP(similarities, all_db, all_videos=True, clean=False)
        if verbose:
            print('-' * 25)
            print('All dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(mAP, mAP_star))

        mAP_c = self.calculate_mAP(similarities, all_db, all_videos=False, clean=True)
        mAP_c_star = self.calculate_mAP(similarities, all_db, all_videos=True, clean=True)
        if verbose:
            print('Clean dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(mAP_c, mAP_c_star))
        return {'mAP': mAP, 'mAP_star': mAP_star, 'mAP_c': mAP_c, 'mAP_c_star': mAP_c_star}
