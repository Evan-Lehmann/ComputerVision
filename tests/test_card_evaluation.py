import copy
import unittest
from evaluate_card_result import evaluate


class CardEvaluationTests(unittest.TestCase):
    def fixture(self):
        cards=[dict(id='low',rank='6',round_number=1),dict(id='high',rank='K',round_number=1)]
        observations=[dict(frame=f,region='player',box=[i*30,0,i*30+15,20],identity=card['id'])
                      for f in [0,1] for i,card in enumerate(cards)]
        reference=dict(scope='synthetic fully annotated example',cards=cards,observations=observations,count_change=0)
        events=[dict(id=str(i),rank=card['rank'],region='player',round_number=1,
                     observation_frames=[0,1],observation_boxes=[[i*30,0,i*30+15,20]]*2)
                for i,card in enumerate(cards)]
        return dict(events=events,count_change=0),reference

    def test_correct_card_identities_and_count_pass(self):
        result,reference=self.fixture();report=evaluate(result,reference)
        self.assertTrue(report['exact_annotated_card_list'])
        self.assertTrue(report['count_matches'])
        self.assertEqual(report['matched_observations'],4)

    def test_canceling_duplicate_ranks_fail_even_when_final_count_matches(self):
        result,reference=self.fixture();events=[]
        for event in result['events']:
            for frame,box in zip(event['observation_frames'],event['observation_boxes']):
                copy=event.copy();copy.update(id=event['id']+'-'+str(frame),
                    observation_frames=[frame],observation_boxes=[box]);events.append(copy)
        result['events']=events;report=evaluate(result,reference)
        self.assertTrue(report['count_matches'])
        self.assertFalse(report['exact_annotated_card_list'])
        self.assertEqual(report['extra_identity_fragments'],2)

    def test_identity_switch_wrong_rank_and_round_are_separate_failures(self):
        result,reference=self.fixture()
        result['events'][0]['observation_boxes'][1]=[30,0,45,20]
        result['events'][1]['observation_boxes'][1]=[0,0,15,20]
        report=evaluate(result,reference)
        self.assertEqual(report['mixed_identity_tracks'],2)
        self.assertFalse(report['exact_annotated_card_list'])
        result,reference=self.fixture();result['events'][0].update(rank='5',round_number=2)
        report=evaluate(result,reference)
        self.assertEqual(report['wrong_accepted_rank_events'],1)
        self.assertEqual(report['wrong_round_events'],1)
        self.assertFalse(report['exact_annotated_card_list'])

    def test_unknowns_missing_observations_and_bad_provenance_fail(self):
        result,reference=self.fixture();result['events'][0]['rank']='UNKNOWN'
        self.assertEqual(evaluate(result,reference)['unknown_events'],1)
        result,reference=self.fixture();result['events'].pop()
        self.assertEqual(evaluate(result,reference)['missing_physical_cards'],1)
        self.assertEqual(evaluate(result,reference)['missing_annotated_observations'],2)
        result,reference=self.fixture();result['events'][0]['observation_boxes']=[]
        with self.assertRaises(ValueError):evaluate(result,reference)


if __name__=='__main__':unittest.main()
