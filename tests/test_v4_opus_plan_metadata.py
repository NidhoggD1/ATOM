import numpy as np

from atom.model_ops.attentions.deepseek_v4_attn import _chunk_cu_seqlens


def test_chunk_cu_seqlens_cuts_sequences_and_drops_empty_segments():
    cu = np.asarray([0, 3, 8, 8, 12], dtype=np.int32)

    assert _chunk_cu_seqlens(cu, 2, 10).tolist() == [0, 1, 6, 8]
