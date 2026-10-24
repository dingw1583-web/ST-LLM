import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from paired_data import paired_segments, grouped_split, temporal_rows
from compare_runs import compare, holm
from config import get_config
from evaluation import metrics


class ReproductionTests(unittest.TestCase):
    def test_common_packet_filter_preserves_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flow.npz'
            headers=np.empty(3,dtype=object);payloads=np.empty(3,dtype=object)
            headers[:]=[list(range(40)),list(range(40)),list(range(40))]
            payloads[:]=[[11],[],[33]]
            np.savez(path,header=headers,payload=payloads)
            records=paired_segments(path,get_config('iscx-vpn'))
            _,header,payload,indices=records[0]
            self.assertEqual(indices,[0,2])
            self.assertEqual(payload[:2,0].tolist(),[11,33])
            self.assertEqual(header[0,:16].tolist(),list(range(12))+list(range(24,28)))
            self.assertTrue((payload[2:]==256).all())

    def test_unpaired_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flow.npz'
            np.savez(path,header=np.zeros((2,40)),payload=np.zeros((1,150)))
            with self.assertRaises(ValueError):
                paired_segments(path,get_config('iscx-vpn'))

    def test_tor_windows_include_final_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flow.npz'
            np.savez(path,header=np.ones((3,40)),payload=np.ones((3,150)),time=[0,60,125])
            records=paired_segments(path,get_config('iscx-tor'),temporal=True)
            self.assertEqual([r[0] for r in records],[0,1,2])
            self.assertEqual([r[3] for r in records],[[0],[1],[2]])

    def test_grouped_split_deterministic_and_disjoint(self):
        groups=['a','a','b','c','d','d']
        train,val=grouped_split(groups,32)
        self.assertEqual((train,val),grouped_split(groups,32))
        self.assertFalse({groups[i] for i in train}&{groups[i] for i in val})
        self.assertEqual(sorted(train+val),list(range(len(groups))))

    def test_temporal_manifest_rejects_leakage_and_unseen_class(self):
        rows=[dict(path='a',label='x',week=1,flow_id='a'),dict(path='b',label='x',week=2,flow_id='b')]
        self.assertEqual(temporal_rows(rows),{'x':0})
        with self.assertRaises(ValueError):
            temporal_rows(rows+[dict(rows[0],week=3)])
        with self.assertRaises(ValueError):
            temporal_rows(rows+[dict(path='c',label='unseen',week=4,flow_id='c')])

    def test_holm_known_example(self):
        np.testing.assert_allclose(holm([.03,.01,.04]),[.06,.03,.06])

    def test_paired_difference_and_zero_variance(self):
        def row(seed,score):
            return dict(dataset='x',split='test',seed=seed,sample_ids=['a'],labels=[0],train_sample_ids=['t'],validation_sample_ids=['v'],sample_group_ids=['ahash'],train_group_ids=['thash'],validation_group_ids=['vhash'],metrics={'accuracy':score})
        a={i:row(i,.8) for i in range(10)}; b={i:row(i,.7) for i in range(10)}
        result=compare(a,b,'accuracy')
        self.assertAlmostEqual(result['mean_difference'],.1)
        self.assertEqual(result['p_value'],0)
        b[0]['sample_ids']=['different']
        with self.assertRaises(ValueError):
            compare(a,b,'accuracy')

    def test_metrics_report_macro_and_weighted(self):
        result=metrics([0,0,1],[0,0,0])
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertAlmostEqual(result['macro_f1'],.4)
        self.assertIn('weighted_f1',result)

    def test_cesnet_config_routes_weeks(self):
        cfg=get_config('cesnet-quic22')
        self.assertEqual(cfg.NUM_CLASSES,75)
        self.assertTrue(cfg.HEADER_TRAIN_DATA.endswith('week1.npz'))
        self.assertTrue(cfg.HEADER_VAL_DATA.endswith('week2.npz'))
        self.assertTrue(cfg.HEADER_TEST_DATA.endswith('week3.npz'))
        self.assertTrue(cfg.HEADER_WEEK4_DATA.endswith('week4.npz'))


if __name__=='__main__':
    unittest.main()

class GraphConstructionTests(unittest.TestCase):
    def test_padding_and_isolated_bytes_have_nodes(self):
        import ast
        import math
        from types import SimpleNamespace
        source=ast.parse((Path(__file__).resolve().parents[1]/'utils.py').read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='construct_graph')
        class Graph:
            def __init__(self,edges,num_nodes):
                self.edges=edges;self.num_nodes=num_nodes;self.ndata={}
        namespace={'math':math,'dgl':SimpleNamespace(graph=Graph,add_self_loop=lambda g:g),
                   'torch':SimpleNamespace(tensor=lambda x,dtype:np.asarray(x,dtype=dtype),float32=np.float32)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'utils.py','exec'),namespace)
        graph=namespace['construct_graph']([256]*150,5)
        self.assertEqual(graph.num_nodes,1)
        self.assertEqual(graph.ndata['feat'].shape,(1,1))
        self.assertEqual(graph.ndata['feat'][0,0],256)

class PreprocessingManifestTests(unittest.TestCase):
    def test_temporal_routing_and_serializable_packet_manifest(self):
        import argparse
        import csv
        from unittest.mock import patch
        import preprocess
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            rows=[]
            for label in range(2):
                for week in range(1,5):
                    name=f'{label}_{week}.npz'
                    np.savez(base/name,header=np.ones((1,40),dtype=int),payload=np.array([[label,week]],dtype=int))
                    rows.append(dict(path=name,label=str(label),week=week,flow_id=f'{label}_{week}'))
            with (base/'input.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['path','label','week','flow_id']);writer.writeheader();writer.writerows(rows)
            cfg=get_config('cesnet-quic22');cfg.NUM_CLASSES=2
            recorded={}
            def capture(header_path,graph_path,records,cfg):
                recorded[Path(header_path).stem]=records
            with patch.object(preprocess,'get_config',return_value=cfg),patch.object(preprocess,'DATA_ROOT',base),patch.object(preprocess,'write_split',side_effect=capture):
                preprocess.prepare(argparse.Namespace(dataset='cesnet-quic22',manifest=base/'input.csv'))
            ledger=json.loads((base/'CESNET-QUIC22'/'split_manifest.json').read_text())
            self.assertEqual(len(ledger),8)
            self.assertEqual(set(recorded),{'week1','week2','week3','week4'})
            self.assertTrue(all(r['packet_indices']==[0] for r in ledger))
            self.assertTrue(all(len(recorded[week])==2 for week in recorded))
