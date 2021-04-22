import os.path as osp

from ..utils.data import Dataset
from ..utils.serialization import write_json
from ..utils.dist_utils import synchronize


class Demo(Dataset):

    def __init__(self, root, scale=None, verbose=True):
        super(Demo, self).__init__(root)

        self.arrange()
        self.load(verbose)

    def arrange(self):
        if self._check_integrity():
            return

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        raw_dir = osp.join(self.root, 'raw')
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")

        meta = {
                'name': 'demo', 
                'identities': identities,
                'utm': utms
                }

        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta.json'))

        splits = {
            'q_train': sorted(q_train_pids),
            'db_train': sorted(db_train_pids),
            'q_val': sorted(q_val_pids),
            'db_val': sorted(db_val_pids),
            'q_test': sorted(q_test_pids),
            'db_test': sorted(db_test_pids)}

        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits.json'))

        synchronize()
