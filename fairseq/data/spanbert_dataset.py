# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
from collections import Counter
from fairseq.data import (
    data_utils,
    FairseqDataset
)
import time
import json
from fairseq.data.masking import (
    ParagraphInfo,
    BertRandomMaskingScheme,
    PairWithSpanMaskingScheme
)


def collate_2d(values, pad_idx, left_pad, move_eos_to_beginning=False):
    size_0 = max(v.size(0) for v in values)
    size_1 = max(v.size(1) for v in values)
    res = values[0].new(len(values), size_0, size_1).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i, size_0 - v.size(0):, size_1 - v.size(1):] if left_pad else res[i, :v.size(0), :v.size(1)])
    return res


class SpanBertDataset(FairseqDataset):
    """
    A wrapper around BlockDataset for BERT data.
    Args:
        dataset (BlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, sizes, vocab, shuffle, seed, args=None):
        self.dataset = dataset
        self.sizes = sizes
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.masking_schemes = []
        self.paragraph_info = ParagraphInfo(vocab)
        self.args = args

        if args.masking_scheme == 'random':
            self.masking_scheme = BertRandomMaskingScheme(
                args,
                self.vocab.symbols,
                self.vocab.count,
                self.vocab.pad(),
                self.vocab.mask()
            )
        elif args.masking_scheme == 'span':
            self.masking_scheme = PairWithSpanMaskingScheme(
                args,
                self.vocab.symbols,
                self.vocab.count,
                self.vocab.pad(),
                self.vocab.mask(),
                self.paragraph_info
            )


    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            block = self.dataset[index]

        masked_block, masked_tgt, pair_targets = self._mask_block(block)

        item = np.concatenate([
            [self.vocab.bos()],
            masked_block,
            [self.vocab.eos()],
        ])
        target = np.concatenate([[self.vocab.pad()], masked_tgt, [self.vocab.pad()]])
        seg = np.zeros(len(block) + 2)
        if pair_targets is not None and len(pair_targets) > 0:
            # dummy = [[0 for i in range(self.args.max_pair_targets + 2)]]
            # add 1 to the first two since they are input indices. Rest are targets.
            pair_targets = [[(x+1) if i < 2 else x for i, x in enumerate(pair_tgt)] for pair_tgt in pair_targets]
            # pair_targets = dummy + pair_targets
            pair_targets = torch.from_numpy(np.array(pair_targets)).long()
        else:
            pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        return {
            'id': index,
            'source': torch.from_numpy(item).long(),
            'segment_labels': torch.from_numpy(seg).long(),
            'lm_target': torch.from_numpy(target).long(),
            'pair_targets': pair_targets,
        }

    def __len__(self):
        return len(self.dataset)

    def _collate(self, samples, pad_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )

        def merge_2d(key):
            return collate_2d(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        pair_targets = merge_2d('pair_targets')

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
                'pairs': pair_targets[:, :, :2]
            },
            'lm_target': merge('lm_target'),
            'nsentences': samples[0]['source'].size(0),
            'pair_targets': pair_targets[:, :, 2:]
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return self._collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=12):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        segment_labels = torch.zeros(tgt_len, dtype=torch.long)
        pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        lm_target = source
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target,
                'pair_targets': pair_targets
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def _mask_block(self, sentence):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        return self.masking_scheme.mask(sentence)
