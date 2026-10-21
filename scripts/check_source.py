import ast
import importlib.util
import io
from pathlib import Path
import tokenize

root = Path(__file__).resolve().parents[1]
for path in root.rglob('*.py'):
    if '.git' in path.parts:
        continue
    source = path.read_text(encoding='utf-8')
    ast.parse(source, filename=str(path))
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)):
        raise AssertionError(f'Code comment found: {path}')

spec = importlib.util.spec_from_file_location('release_config', root/'config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
for dataset, cls in config.DATASET_CONFIGS.items():
    assert cls.NUM_CLASSES == len(cls.CLASS_NAMES) == len(cls.DIR_PATH_DICT), dataset
    for label, name in enumerate(cls.CLASS_NAMES):
        assert Path(cls.DIR_PATH_DICT[label]).name == name
        prefix = dataset.replace('iscx', 'ISCX').replace('ustc-tfc2016', 'USTC-TFC2016')
        directory = root/'DATA'/Path(cls.TRAIN_DATA).parent.name
        expected = list(directory.glob('DATA_*_'+name+'.txt'))
        assert len(expected) == 1, (dataset, name)
assert config.USTCTFC2016Config.NUM_CLASSES == 20
assert config.USTCTFC2016Config.CLASS_NAMES[10] == 'Malware_Cridex'
assert config.USTCTFC2016Config.CLASS_NAMES[19] == 'Malware_Zeus'
print('Source syntax, comment removal, class counts and categorization mappings passed.')

assert len(config.Config.SEEDS) == len(set(config.Config.SEEDS)) == 10
assert config.Config.SEEDS[0] == 32
assert config.USTCTFC2016Config.SEEDS == config.Config.SEEDS
assert config.USTCTFC2016Config.SEED == 32
train_source = (root/'train.py').read_text(encoding='utf-8')
assert 'manual_seed(42)' not in train_source
assert 'set_seed(42)' not in train_source
assert train_source.count('manual_seed(args.seed)') == 2
print('Ten-seed configuration and training seed wiring passed.')
