# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq import utils
from fairseq.data import (
    Dictionary,
    data_utils,
    TokenBlockDataset,
    TruncateDataset,
)
from fairseq.data.spanbert_dataset import SpanBertDataset
from fairseq.tasks import (
    FairseqTask,
    register_task
)
from fairseq.data.masking import ParagraphInfo
import logging 

logger = logging.getLogger(__name__)


class BertDictionary(Dictionary):
    """Dictionary for BERT tasks
        extended from Dictionary by adding support for cls as well as mask symbols"""
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        cls='[CLS]',
        mask='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad=pad, unk=unk)
        (
            self.cls_word,
            self.mask_word,
            self.sep_word,
        ) = cls, mask, sep

        self.is_end = None
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        idx = self.index(self.mask_word)
        return idx

    def is_end_word(self, idx):
        if self.is_end is None:
            self.is_end = [self.symbols[i].endswith("</w>") for i in range(len(self))]
        return self.is_end[idx]

@register_task('span_bert')
class SpanBertTask(FairseqTask):
    """
    Train BERT model.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments'
                                 ' per sample for BERT dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--break-mode', default="complete", type=str, help='mode for breaking sentence')
        parser.add_argument('--masking-scheme', default='random', type=str, help='chosen masking scheme')
        parser.add_argument('--span-lower', default=1, type=int, help='lower bound on the number of words in a span')
        parser.add_argument('--span-upper', default=10, type=int, help='upper bound on the number of words in a span')
        parser.add_argument('--max-pair-targets', default=20, type=int, help='max word pieces b/w a pair')
        parser.add_argument('--mask-ratio', default=0.15, type=float, help='proportion of words to be masked')
        parser.add_argument('--geometric-p', default=0.2, type=float, help='p for the geometric distribution used in span masking. -1 is uniform')
        parser.add_argument('--pair-loss-weight', default=1.0, type=float, help='weight for pair2/SBO loss')
        parser.add_argument('--short-seq-prob', default=0.0, type=float)
        parser.add_argument('--pair-target-layer', default=-1, type=int)
        parser.add_argument('--pair-positional-embedding-size', default=200, type=int)
        parser.add_argument('--ner-masking-prob', default=0.5, type=float)
        parser.add_argument('--replacement-method', default='span')
        parser.add_argument('--return-only-spans', default=False, action='store_true')
        parser.add_argument('--shuffle-instance', default=False, action='store_true')
        parser.add_argument('--endpoints', default='external', type=str)
        parser.add_argument('--skip-validation', default=False, action='store_true')
 
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.args = args
        self.dictionary = dictionary
        args.vocab_size = len(dictionary)
        self.seed = args.seed
        self.short_seq_prob = args.short_seq_prob
        self.mask_idx = dictionary.add_symbol(dictionary.mask_word)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = BertDictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info('| Dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError("Dataset not found: {} ({})".format(split, split_path))

        dataset = TruncateDataset(dataset, self.args.tokens_per_sample - 2)

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 2,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.break_mode
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        self.datasets[split] = SpanBertDataset(
            dataset, dataset.sizes, self.dictionary,
            shuffle=self.args.shuffle_instance, seed=self.seed, args=self.args
        )
