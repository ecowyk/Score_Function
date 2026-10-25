"""Protocol regressions for the two-stage Energy run; no GPU/data required."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_energy_aligned_1025 as prepare
import run_energy_aligned_1025 as runner


def source_fixture():
    config=json.loads((ROOT/'configs/score_function.json').read_text())
    config['model'].update(parameterization='score',temporal_attention=True,neighbor_future=False)
    config['training'].update(sigma=0.1,batch_size=2048,microbatch_size=256,
        early_stopping=False,lr_schedule='constant_after_warmup',
        data_augmentation=dict(enabled=True,probability=0.5,encoding_batch_size=64))
    config['output']='source/run'; config['paths']['run_dir']='source/run'
    info=dict(world_size=8,global_batch=2048,microbatch_per_rank=256,gradient_accumulation=1,
        train_samples=947912,precision='float32',training_budget=dict(planned_epochs=529,
        planned_optimizer_updates=244398))
    status=dict(state='complete',epoch=529,step=244398)
    return config,info,status


class EnergyProtocolTests(unittest.TestCase):
    def test_energy_preserves_training_seed_budget_and_every_non_output_field(self):
        original,info,status=source_fixture()
        before=copy.deepcopy(original)
        prepare.verify_source_metadata(original,info,status)
        changed=prepare.energy_config(original,Path('/new/run'))
        self.assertEqual(original,before)
        self.assertEqual(changed['training'],original['training'])
        self.assertEqual(changed['data'],original['data'])
        self.assertEqual(changed['reference_seed'],original['reference_seed'])
        self.assertNotEqual(changed['model']['parameterization'],original['model']['parameterization'])
        restored=copy.deepcopy(changed)
        restored['model']['parameterization']='score'
        restored['output']=original['output']; restored['paths']['run_dir']=original['paths']['run_dir']
        self.assertEqual(restored,original)

    def test_reject_short_training_or_wrong_batch_or_old_energy_config(self):
        mutations=(('status','step',1000),('info','world_size',4),('info','microbatch_per_rank',64),
                   ('info','precision','float16'))
        for object_name,key,value in mutations:
            with self.subTest(key=key):
                config,info,status=source_fixture()
                (status if object_name=='status' else info)[key]=value
                with self.assertRaises(ValueError): prepare.verify_source_metadata(config,info,status)
        config,info,status=source_fixture(); config['training']['sigma']=0.05
        with self.assertRaises(ValueError): prepare.verify_source_metadata(config,info,status)
        config,info,status=source_fixture(); config['model']['parameterization']='energy'
        with self.assertRaises(ValueError): prepare.verify_source_metadata(config,info,status)

    def test_single_seed_five_positive_Ks_and_reused_zero_control(self):
        jobs=runner.evaluation_jobs('hard_NR')
        self.assertEqual([j['K'] for j in jobs],[0,5,10,20,50,100])
        self.assertEqual({j['seed'] for j in jobs},{20260922})
        self.assertEqual(jobs[0]['id'],'hard_NR_seed20260922_K0')
        self.assertEqual(jobs[0]['model'],'DP_K0_reused')
        self.assertEqual({j['model'] for j in jobs[1:]},{'E10-G_seed20260922'})
        self.assertEqual({j['split'] for j in runner.evaluation_jobs('random_NR')},{'test14-random'})

    def test_borrow_control_requires_original_validator_and_identical_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); job=dict(id='hard_NR_seed20260922_K0',model='S10-G_seed20260922')
            plan=dict(jobs=[job]); receipt=dict(job,state='complete',result_dir='original/result',
                metrics=dict(score=0.75),scenario_keys=[['log','token']],num_scenarios=1)
            prepare.suite.write(out/'jobs'/f"{job['id']}.json",receipt)
            checked={key:receipt[key] for key in ('metrics','scenario_keys','num_scenarios')}
            with patch.object(prepare.matrix,'validate_result',return_value=checked) as validate:
                _,actual=prepare.validate_control(out,plan,'hard_NR')
                self.assertEqual(actual,receipt)
                validate.assert_called_once_with('original/result',job,plan,out)
            checked['metrics']=dict(score=0.76)
            with patch.object(prepare.matrix,'validate_result',return_value=checked):
                with self.assertRaises(ValueError): prepare.validate_control(out,plan,'hard_NR')

    def test_failed_training_never_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); prepare.suite.write(out/'plan.json',dict(jobs=[]))
            args=SimpleNamespace(queue_dir=out,poll_seconds=1)
            with patch.object(runner.training,'verify_plan'), \
                 patch.object(runner,'run_gpu_smoke'), \
                 patch.object(runner.training,'run',side_effect=RuntimeError('training failed')), \
                 patch.object(runner,'run_evaluations') as evaluate:
                with self.assertRaisesRegex(RuntimeError,'training failed'): runner.run(args)
                evaluate.assert_not_called()

    def test_complete_training_hands_off_to_evaluation_same_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); plan=dict(jobs=[]); prepare.suite.write(out/'plan.json',plan)
            args=SimpleNamespace(queue_dir=out,poll_seconds=1)
            with patch.object(runner.training,'verify_plan'), \
                 patch.object(runner,'run_gpu_smoke'), \
                 patch.object(runner.training,'run',return_value=0), \
                 patch.object(runner,'run_evaluations') as evaluate:
                runner.run(args)
                evaluate.assert_called_once_with(out.resolve(),plan,1)
                self.assertEqual(prepare.suite.read(out/'pipeline_status.json')['state'],'complete')

    def test_repeated_prepare_verifies_same_queue_and_rejects_changed_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'energy'
            source=root/'source'; matrix_out=root/'matrix'
            plan=dict(root=str(root),source_queue=str(source),evaluation=dict(matrix=str(matrix_out),
                setting='hard_NR',gpus=list(range(8)),seed=20260922,ks=[5,10,20,50,100]))
            prepare.suite.write(out/'plan.json',plan)
            args=SimpleNamespace(root=root,queue_dir=out,source_queue=source,matrix=matrix_out,
                setting='hard_NR',gpus='0,1,2,3,4,5,6,7')
            with patch.object(prepare.training,'verify_plan') as verify:
                self.assertEqual(prepare.prepare(args),plan)
                verify.assert_called_once_with(plan)
            args.setting='random_NR'
            with self.assertRaises(ValueError): prepare.prepare(args)


if __name__=='__main__':
    unittest.main()
