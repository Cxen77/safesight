from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from signals import associate_ppe, normalize_inputs, FireHistory


class SignalTests(unittest.TestCase):
    def test_user_approved_normalization_and_empty_scene(self):
        values=normalize_inputs(2,1,2,1,5,.25,.4)
        self.assertAlmostEqual(values['ppe_load'],2/3)
        self.assertEqual(values['zone_pressure'],.75)
        self.assertEqual(values['proximity'],.75)
        self.assertEqual(normalize_inputs(0,0,0,0,0,1,.8),
                         dict(ppe_load=0,zone_pressure=0,proximity=0,fire_evidence=.8))
        with self.assertRaises(ValueError):
            normalize_inputs(1,2,0,0,0,1,0)

    def test_carried_helmet_does_not_cover_a_bare_head(self):
        persons=[{'id':1,'box':(0,0,100,300)}]
        equipment=[{'name':'head','box':(20,5,80,60)}, {'name':'helmet','box':(20,240,80,290)}]
        result=associate_ppe(persons,equipment)[0]
        self.assertTrue(result['bare_head'])
        self.assertFalse(result['helmet'])
        self.assertFalse(associate_ppe(persons,[])[0]['bare_head'])

    def test_one_item_cannot_cover_two_overlapping_people(self):
        persons=[{'id':1,'box':(0,0,100,300)}, {'id':2,'box':(20,0,120,300)}]
        result=associate_ppe(persons,[{'name':'helmet','box':(30,5,70,50)}])
        self.assertEqual(sum(p['helmet'] for p in result),1)

    def test_fire_history_is_max_of_ten_checks(self):
        history=FireHistory()
        history.add(.9)
        for _ in range(9): history.add(0)
        self.assertEqual(history.evidence,.9)
        history.add(0)
        self.assertEqual(history.evidence,0)
        with self.assertRaises(ValueError): history.add(float('nan'))


if __name__=='__main__':
    unittest.main()
