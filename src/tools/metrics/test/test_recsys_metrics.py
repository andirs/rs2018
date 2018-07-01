import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
from recsys_metrics import r_precision, get_relevance, dcg, idcg, ndcg, rsc


class RecSysTest(unittest.TestCase):
    def setUp(self):
        self.ground_truth = ['1', '2', '3', '5', '8', '99']
        self.prediction = ['5', '8', '13', '3']

        self.ground_truth_rsc_one = [1]
        self.ground_truth_rsc_two = [499]
        self.ground_truth_rsc_three = [500]
        self.prediction_rsc = range(500)

    def test_r_precision(self):
        r_precision_score = r_precision(self.ground_truth, self.prediction)[1]
        r_precision_list = r_precision(self.ground_truth, self.prediction)[0]
        self.assertEqual(r_precision_score, 0.5)
        self.assertEqual(r_precision_list, ['5', '8', '3'])

    def test_get_relevance(self):
        test_item_one = '1'
        test_item_two = '42'
        test_item_three = None
        self.assertEqual(get_relevance(
            self.ground_truth, test_item_one), 1)
        self.assertEqual(get_relevance(
            self.ground_truth, test_item_two), 0)
        self.assertEqual(get_relevance(
            self.ground_truth, test_item_three), 0)

    def test_dcg(self):
        dcg_score = dcg(self.ground_truth, self.prediction)
        self.assertEqual(2.5, dcg_score)

    def test_idcg(self):
        idcg_score = idcg(self.ground_truth, self.prediction)
        self.assertAlmostEqual(2.6309297535714575, idcg_score)

    def test_ndcg(self):
        ndcg_score = ndcg(self.ground_truth, self.prediction)
        self.assertAlmostEqual(0.9502344167898358, ndcg_score)

    def test_rsc(self):
        rsc_one = rsc(
            self.ground_truth_rsc_one, self.prediction_rsc)
        rsc_two = rsc(
            self.ground_truth_rsc_two, self.prediction_rsc)
        rsc_three = rsc(
            self.ground_truth_rsc_three, self.prediction_rsc)
        self.assertEqual(1, rsc_one)
        self.assertEqual(50, rsc_two)
        self.assertEqual(51, rsc_three)



if __name__ == '__main__':
    unittest.main()
