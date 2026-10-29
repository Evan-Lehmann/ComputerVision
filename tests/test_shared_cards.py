"""Cross-view identity, ambiguity and coverage gates using synthetic evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
import torch
from shared_cards import SharedMonitor,DEFAULT_SHARED_VIEW,mini_hand
from card_suits import SuitReader,suit_tensor
from round_pipeline import summarize,apply_review,validate_shared_view,DEFAULT_REGIONS


def primary(rank,suit,i,first=0,round_number=1):
    return dict(id=str(i),region='player',rank=rank,suit=suit,round_number=round_number,
        first_frame=first,last_frame=first+20,observation_frames=list(range(first,first+21)),
        observations=21,evidence=[],excluded=False,reviewed=False)


def snapshots(third=('9','hearts'),offset=0):
    result=[]
    for count,rank,suit,frames in [(1,'9','clubs',[0,1]),(2,'4','spades',[2,3]),(3,*third,[4,5])]:
        for f in frames:
            image=np.full((20,20,3),255,np.uint8);cv2.putText(image,rank,(1,16),0,.5,(0,0,0),1)
            result.append(dict(count=count,frame=f+offset,source=f'frame_{f+offset}.png',crop=image,
                hand_crop=image,box=[10,10,20,20],face_box=[10,10,40,40],suit_crops=[image],
                test_rank=rank,test_suit=suit))
    return result


class RankReader:
    def consensus(self,obs):
        if len({o['frame'] for o in obs})<2:return None
        labels={o['test_rank'] for o in obs}
        return dict(rank=next(iter(labels)),score=.95) if len(labels)==1 and 'UNKNOWN' not in labels else None


class Suits:
    def consensus(self,obs):
        if len({o['frame'] for o in obs})<2:return None
        labels={o['test_suit'] for o in obs}
        return dict(suit=next(iter(labels)),score=.95) if len(labels)==1 and 'UNKNOWN' not in labels else None


class SharedIdentityTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def resolve(self,events,shots=None):
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits());monitor.snapshots={1:snapshots() if shots is None else shots}
        return monitor.resolve(events,self.root)
    def base(self):return [primary('9','clubs',1),primary('4','spades',2,2)]
    def test_shared_prefix_is_linked_once_and_distinct_neutral_nine_is_kept(self):
        events=self.base();c=self.resolve(events)
        self.assertEqual(c['status'],'configured_views_reconciled')
        self.assertEqual(len(c['links']),2);self.assertEqual(len(events),3)
        self.assertEqual((events[-1]['rank'],events[-1]['suit']),('9','hearts'))
        self.assertEqual(events[-1]['identity_status'],'distinct')
        self.assertEqual(events[0]['source_views'],['central','player'])
    def test_same_rank_and_suit_can_still_be_a_distinct_new_card(self):
        events=self.base();c=self.resolve(events,snapshots(('9','clubs')))
        self.assertEqual(len(events),3)
        self.assertEqual(c['slots'][-1]['status'],'additional')
        self.assertEqual(events[-1]['rank'],'9')
    def test_identical_tuple_in_an_unlinked_dealer_hand_is_ambiguous(self):
        dealer=primary('9','hearts',3,0);dealer['region']='dealer'
        events=self.base()+[dealer];c=self.resolve(events)
        self.assertEqual(c['slots'][-1]['status'],'needs_review')
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertIn('3',events[-1]['possible_duplicates'])
    def test_different_rank_in_unknown_suit_does_not_compete_with_a_known_extra(self):
        dealer=primary('K','UNKNOWN',3,0);dealer['region']='dealer'
        events=self.base()+[dealer];c=self.resolve(events)
        self.assertEqual(c['slots'][-1]['status'],'additional')
    def test_coherent_bounded_view_delay_can_link_a_prefix(self):
        shots=snapshots()[:4]+[dict(snapshots()[3],frame=f) for f in [4,5,6,7,8]]
        events=[primary('9','clubs',1,4),primary('4','spades',2,6)]
        c=self.resolve(events,shots)
        self.assertEqual(len(c['links']),2)

    def late_shared_append(self):
        # The main hand ends at frame22. Its shared representation remains
        # visible through frame40 before a third card arrives at frame41.
        shots=snapshots()[:4]
        shots += [dict(shots[-1],frame=f) for f in range(4,41)]
        shots += [dict(o,frame=f) for o,f in zip(snapshots()[4:],[41,42])]
        for shot in shots:
            count=shot['count'];shot['hand_box']=[720-count*4,633,760+count*4,665]
            patches=[]
            for label in ['9','4','9'][:count]:
                patch=np.full((18,12,3),255,np.uint8)
                cv2.putText(patch,label,(1,14),0,.5,(0,0,0),1)
                patches.append(patch)
            shot['prefix_patches']=patches
        return shots

    def test_observed_shared_continuation_keeps_new_card_after_main_disappears(self):
        events=self.base();c=self.resolve(events,self.late_shared_append())
        self.assertEqual(len(c['links']),2)
        self.assertEqual(c['status'],'configured_views_reconciled')
        self.assertEqual(events[-1]['rank'],'9')
        self.assertEqual(events[-1]['identity_basis'],'continuous_shared_hand')

    def test_late_same_rank_suit_as_prefix_still_has_separate_identity(self):
        shots=self.late_shared_append()
        for s in shots:
            if s['count']==3:s['test_suit']='clubs'
        events=self.base();self.resolve(events,shots)
        self.assertEqual(events[-1]['rank'],'9')
        self.assertEqual(events[-1]['identity_status'],'distinct')
        self.assertNotEqual(events[-1]['id'],events[0]['id'])

    def test_long_gap_cannot_inherit_an_old_shared_identity(self):
        shots=[s for s in self.late_shared_append() if s['frame']<5 or s['frame']>40]
        events=self.base();c=self.resolve(events,shots)
        self.assertEqual(events[-1]['identity_status'],'needs_review')
        self.assertEqual(c['slots'][-1]['rank'],'9')
        self.assertEqual(events[-1]['rank'],'UNKNOWN')

    def test_replaced_prefix_or_large_move_cannot_inherit_shared_identity(self):
        for corruption in ['glyph','colour','position','blank','missing']:
            shots=self.late_shared_append()
            for s in shots:
                if s['count']!=3:continue
                if corruption=='glyph':s['prefix_patches'][0]=s['prefix_patches'][1]
                if corruption=='colour':s['prefix_patches'][0][:,:,2]=255
                if corruption=='position':s['hand_box']=[800,633,850,665]
                if corruption=='blank':s['prefix_patches'][0][:]=255
                if corruption=='missing':s.pop('prefix_patches')
            events=self.base();self.resolve(events,shots)
            self.assertEqual(events[-1]['identity_status'],'needs_review',corruption)

    def test_one_bad_frame_does_not_break_an_otherwise_visible_continuation(self):
        shots=self.late_shared_append()
        shots[25]['prefix_patches']=[np.full_like(p,255) for p in shots[25]['prefix_patches']]
        events=self.base();self.resolve(events,shots)
        self.assertEqual(events[-1]['identity_status'],'distinct')

    def test_continuation_still_rejects_competing_dealer_identity(self):
        dealer=primary('9','hearts',3);dealer['region']='dealer'
        events=self.base()+[dealer];self.resolve(events,self.late_shared_append())
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertIn('3',events[-1]['possible_duplicates'])

    def test_a_single_late_frame_cannot_confirm_additional_identity(self):
        events=self.base();self.resolve(events,self.late_shared_append()[:-1])
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertEqual(events[-1]['identity_status'],'needs_review')

    def test_continuation_requires_repeated_joint_visibility_of_the_original_prefix(self):
        events=self.base()
        events[0]['observation_frames']=[0,1,2]
        events[1]['observation_frames']=[2,3,4]
        c=self.resolve(events,self.late_shared_append())
        self.assertEqual(len(c['links']),2)
        self.assertEqual(events[-1]['identity_status'],'needs_review')

    def test_no_extra_shared_card_does_not_create_rows(self):
        events=self.base();c=self.resolve(events,snapshots()[:4])
        self.assertEqual(len(events),2);self.assertEqual(len(c['links']),2)
    def test_birth_mismatch_prevents_equal_tuple_from_being_used_as_global_identity(self):
        events=self.base()+[primary('9','hearts',3,12)]
        c=self.resolve(events)
        self.assertEqual(c['slots'][-1]['status'],'needs_review')
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertEqual(len(c['links']),2)
    def test_a_single_matching_anchor_is_insufficient(self):
        events=[primary('9','clubs',1)];c=self.resolve(events,snapshots()[:2])
        self.assertEqual(c['links'],[]);self.assertEqual(events[-1]['rank'],'UNKNOWN')
    def test_unknown_suit_or_rank_cannot_establish_duplicate_prefix(self):
        for field in ['rank','suit']:
            events=self.base();events[0][field]='UNKNOWN';c=self.resolve(events)
            self.assertEqual(c['links'],[])
            self.assertEqual(c['status'],'needs_review')
    def test_brief_card_is_not_silently_dropped(self):
        events=self.base();c=self.resolve(events,snapshots()[:5])
        self.assertEqual(len(events),3)
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertEqual(c['status'],'needs_review')
    def test_already_overlapped_entry_keeps_context_but_not_a_false_rank_training_crop(self):
        events=self.base();c=self.resolve(events,snapshots()[2:])
        first=c['slots'][0]
        self.assertEqual(first['rank'],'UNKNOWN')
        self.assertTrue(first['evidence'])
        self.assertFalse(first['evidence'][0]['rank_crop'])
    def test_stable_shrink_requires_review_instead_of_merging_two_hands(self):
        shots=snapshots()+[dict(o,frame=o['frame']+10) for o in snapshots()[2:4]]
        events=self.base();c=self.resolve(events,shots)
        self.assertEqual(c['links'],[]);self.assertEqual(c['status'],'needs_review')
    def test_rounds_have_separate_link_identities(self):
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits())
        monitor.snapshots={1:snapshots()[:4],2:snapshots(offset=100)[:4]}
        events=self.base()+[primary('9','clubs','a',100,2),primary('4','spades','b',102,2)]
        c=monitor.resolve(events,self.root)
        self.assertEqual(len(events),4)
        self.assertEqual({l['event_id'] for l in c['links'] if l['round_number']==2},{'a','b'})
    def test_unknown_additional_rank_is_kept_even_when_identity_is_distinct(self):
        events=self.base();c=self.resolve(events,snapshots(('UNKNOWN','hearts')))
        self.assertEqual(events[-1]['rank'],'UNKNOWN')
        self.assertEqual(events[-1]['identity_status'],'distinct')
        self.assertEqual(c['status'],'configured_views_reconciled')
    def test_missing_view_does_not_become_a_complete_count(self):
        events=self.base();c=self.resolve(events,[])
        result=summarize(dict(events=events,coverage=c,needs_coverage_review=True))
        self.assertEqual(result['unresolved'],0);self.assertIsNone(result['count_change'])
        self.assertEqual(result['captured_count_change'],1)
        edits=[dict(id=e['id'],rank=e['rank'],suit=e['suit']) for e in events]
        reviewed=apply_review(result,edits,True)
        self.assertFalse(reviewed['count_verified'])
        confirmed=apply_review(result,edits,True,coverage_confirmed=True)
        self.assertTrue(confirmed['count_verified']);self.assertEqual(confirmed['count_change'],1)
    def test_unchecked_earlier_coverage_blocks_later_running_total(self):
        events=self.base()+[primary('A','hearts',3,30,2)]
        result=summarize(dict(events=events,needs_coverage_review=True,coverage=dict(status='needs_review',issues=[dict(round_number=1,reason='missing shared view')],links=[],slots=[])))
        self.assertIsNone(result['rounds'][0]['count_change'])
        self.assertEqual(result['rounds'][1]['count_change'],-1)
        self.assertIsNone(result['rounds'][1]['running_count_change'])
        edits=[dict(id=e['id'],rank=e['rank'],suit=e['suit'],round_number=e['round_number']) for e in events]
        reviewed=apply_review(result,edits,True,rounds_confirmed=True,coverage_confirmed=True)
        self.assertTrue(reviewed['count_verified'])
        self.assertEqual(reviewed['rounds'][1]['running_count_change'],0)

    def test_lost_shared_view_while_primary_cards_remain_blocks_complete_count(self):
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits());monitor.snapshots={1:snapshots()};monitor.primary_frames={1:list(range(10))}
        events=self.base();c=monitor.resolve(events,self.root)
        self.assertEqual(c['status'],'needs_review')
        self.assertTrue(any(i.get('frame')==6 for i in c['issues']))
        result=summarize(dict(events=events,coverage=c,needs_coverage_review=True))
        self.assertEqual(result['unresolved'],0);self.assertIsNone(result['count_change'])
    def test_normal_table_clear_after_last_primary_card_is_not_a_lost_view(self):
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits());monitor.snapshots={1:snapshots()};monitor.primary_frames={1:list(range(6))}
        events=self.base();c=monitor.resolve(events,self.root)
        self.assertEqual(c['status'],'configured_views_reconciled')

    def test_changing_a_linked_main_card_invalidates_automatic_coverage(self):
        events=self.base();c=self.resolve(events,snapshots()[:4]);result=summarize(dict(events=events,coverage=c,needs_coverage_review=False))
        edits=[dict(id=e['id'],rank=e['rank'],suit=e['suit'],excluded=e['id']=='1') for e in events]
        r=apply_review(result,edits,True)
        self.assertTrue(r['needs_coverage_review']);self.assertFalse(r['count_verified'])
        self.assertTrue(apply_review(result,edits,True,coverage_confirmed=True)['count_verified'])
    def test_bad_suit_and_shared_calibration_are_rejected(self):
        result=summarize(dict(events=self.base()))
        with self.assertRaises(ValueError):apply_review(result,[dict(id='1',rank='9',suit='joker'),dict(id='2',rank='4')],True)
        with self.assertRaises(ValueError):validate_shared_view(dict(DEFAULT_SHARED_VIEW,linked_region='missing'),DEFAULT_REGIONS)
        with self.assertRaises(ValueError):validate_shared_view(dict(DEFAULT_SHARED_VIEW,rect=DEFAULT_REGIONS[0]['rect']),DEFAULT_REGIONS)
        self.assertFalse(validate_shared_view(False,DEFAULT_REGIONS))


class AuxiliaryModelTests(unittest.TestCase):
    def test_missing_and_corrupt_auxiliary_models_fail_closed(self):
        from mini_rank import MiniRankReader
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'missing.pt'
            self.assertFalse(SuitReader(path).available)
            self.assertFalse(MiniRankReader(path).available)
            path.write_bytes(b'invalid checkpoint')
            self.assertFalse(SuitReader(path).available)
            self.assertFalse(MiniRankReader(path).available)
    def test_mini_rank_reader_does_not_assign_a_rank_to_blank_paper(self):
        from mini_rank import MiniRankReader
        reader=MiniRankReader()
        self.assertTrue(reader.available)
        self.assertIsNone(reader.consensus([dict(frame=i,crop=np.full((18,16,3),255,np.uint8)) for i in [0,1]]))
    def test_no_card_evidence_never_claims_configured_coverage(self):
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits())
        with tempfile.TemporaryDirectory() as directory:c=monitor.resolve([],directory)
        self.assertEqual(c['status'],'needs_review')
    def test_audited_training_inputs_reject_wrong_labels_checksums_and_paths(self):
        import csv,hashlib
        from train_crossview_models import load_audited
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=np.full((20,20,3),255,np.uint8);cv2.imwrite(str(root/'crop.png'),image)
            row=dict(crop='crop.png',label='hearts',card_group='one-card',kind='suit',sha256=hashlib.sha256((root/'crop.png').read_bytes()).hexdigest())
            def write(rows):
                with (root/'labels.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            write([row]);self.assertEqual(len(load_audited(root)),1)
            for override in [dict(label='joker'),dict(sha256='wrong'),dict(crop='../crop.png')]:
                write([dict(row,**override)])
                with self.assertRaises(ValueError):load_audited(root)
            write([row,dict(row,label='clubs')])
            with self.assertRaises(ValueError):load_audited(root)


class MiniGeometryTests(unittest.TestCase):
    def test_flat_overlapping_mini_cards_have_separate_slots(self):
        for count in range(1,8):
            image=np.full((900,1440,3),(45,65,35),np.uint8)
            left=720-(count-1)*5
            for i in range(count):
                x=left+i*9;image[633:665,x:x+40]=255;image[633:665,x]=215
            hand=mini_hand(image,DEFAULT_SHARED_VIEW['rect'])
            self.assertIsNotNone(hand)
            self.assertEqual(hand['count'],count)
    def test_antialiased_outer_row_does_not_drop_an_intact_four_card_hand(self):
        image=np.full((900,1440,3),(45,65,35),np.uint8)
        for i in range(4):
            x=707+i*9;image[633:666,x:x+40]=255;image[633:666,x]=215
        image[665,707:774]=150
        image[665,750:760]=220  # Component includes a partly white fringe.
        hand=mini_hand(image,DEFAULT_SHARED_VIEW['rect'])
        self.assertIsNotNone(hand)
        self.assertEqual(hand['count'],4)
    def test_oversized_or_clipped_shared_hand_is_reported_as_unsupported(self):
        image=np.full((900,1440,3),(45,65,35),np.uint8)
        for i in range(8):
            x=680+i*9;image[633:665,x:x+40]=255;image[633:665,x]=215
        hand=mini_hand(image,DEFAULT_SHARED_VIEW['rect'])
        self.assertTrue(hand['unsupported'])
        monitor=SharedMonitor(DEFAULT_SHARED_VIEW,RankReader(),Suits());monitor.observe(image,3,'frame.png',1)
        with tempfile.TemporaryDirectory() as directory:c=monitor.resolve([primary('9','clubs',1)],directory)
        self.assertEqual(c['status'],'needs_review')
        self.assertTrue(any(i.get('frame')==3 for i in c['issues']))
    def test_cardvision_window_cards_are_not_the_casino_shared_hand(self):
        image=np.full((900,1440,3),(43,38,32),np.uint8);image[633:665,720:760]=255
        self.assertIsNone(mini_hand(image,DEFAULT_SHARED_VIEW['rect']))

    def test_empty_or_noisy_video_is_not_a_flat_shared_hand(self):
        image=np.full((900,1440,3),45,np.uint8)
        self.assertIsNone(mini_hand(image,DEFAULT_SHARED_VIEW['rect']))
        rng=np.random.default_rng(4);image[633:665,720:760]=rng.integers(150,255,(32,40,3),dtype=np.uint8)
        self.assertIsNone(mini_hand(image,DEFAULT_SHARED_VIEW['rect']))


class SuitEvidenceTests(unittest.TestCase):
    def reader(self,probabilities):
        reader=SuitReader.__new__(SuitReader);reader.available=True;reader.error=None
        reader.model=Mock(return_value=torch.log(torch.tensor([probabilities])))
        return reader
    def red_icon(self,heart=False):
        image=np.full((40,40,3),255,np.uint8);ink=(20,25,190)
        if heart:
            cv2.circle(image,(14,13),7,ink,-1);cv2.circle(image,(26,13),7,ink,-1)
            cv2.fillPoly(image,[np.array([[7,14],[33,14],[20,33]])],ink)
        else:cv2.fillPoly(image,[np.array([[20,5],[34,20],[20,35],[6,20]])],ink)
        return image
    def test_color_gate_rejects_high_confidence_black_suit_for_red_ink(self):
        self.assertFalse(self.reader([.01,.01,.01,.97]).read(self.red_icon())['accepted'])
    def test_diamond_geometry_does_not_turn_heart_into_diamond(self):
        reader=self.reader([.025,.70,.25,.025])
        self.assertTrue(reader.read(self.red_icon())['accepted'])
        self.assertFalse(reader.read(self.red_icon(True))['accepted'])
    def test_blank_and_multiple_icons_abstain(self):
        self.assertIsNone(suit_tensor(np.full((25,25,3),255,np.uint8)))
        image=np.full((25,50,3),255,np.uint8)
        cv2.circle(image,(10,12),7,(0,0,0),-1);cv2.circle(image,(36,12),7,(0,0,0),-1)
        self.assertIsNone(suit_tensor(image))
    def test_distinct_frames_and_no_conflicting_corners_are_required(self):
        reader=self.reader([.01,.97,.01,.01]);reader.read=Mock(return_value=dict(suit='diamonds',score=.97,accepted=True))
        o=dict(frame=0,suit_crops=[self.red_icon()])
        self.assertIsNone(reader.consensus([o,o]))
        self.assertEqual(reader.consensus([o,dict(o,frame=1)])['suit'],'diamonds')
        reader.read=Mock(side_effect=[dict(suit='diamonds',accepted=True,score=.97),dict(suit='hearts',accepted=True,score=.97)])
        self.assertIsNone(reader.consensus([dict(frame=0,suit_crops=[self.red_icon(),self.red_icon(True)]),dict(frame=1,suit_crops=[self.red_icon()])]))


if __name__=='__main__':unittest.main()
