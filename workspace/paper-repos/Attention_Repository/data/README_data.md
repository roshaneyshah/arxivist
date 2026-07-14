# Data Setup — WMT14 EN-DE / EN-FR

## Quick Start (Automated)

```bash
python prepare_data.py --config configs/base.yaml
```

This will:
1. Download WMT14 EN-DE via the HuggingFace `datasets` library (~1.6 GB)
2. Train a shared SentencePiece BPE model (vocab_size=37000)
3. Place processed files in `data/wmt14_en_de/`

Estimated time: ~15–30 minutes. Estimated disk: ~5 GB.

---

## Manual Setup

If automated download fails, place raw files here:

```
data/wmt14_en_de/
├── train.en
├── train.de
├── newstest2013.en      ← validation set
├── newstest2013.de
├── newstest2014.en      ← test set (Table 2 in paper)
└── newstest2014.de
```

Then train the SentencePiece model manually:

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="data/wmt14_en_de/train_combined.txt",
    model_prefix="data/wmt14_en_de/spm",
    vocab_size=37000,
    model_type="bpe",
    ...
)
```

---

## English-French (EN-FR)

For EN-FR (Section 5.1, 36M sentence pairs, 32k word-piece vocab):

1. Edit `configs/base.yaml`:
   - `data.tgt_lang: fr`
   - `data.vocab_size: 32000`
   - `data.data_dir: data/wmt14_en_fr`
   - `data.tokenizer: sentencepiece_bpe`  # word-piece approximated via BPE

2. Re-run `prepare_data.py` with the updated config.

---

## Paper Citation (Section 5.1)

> We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
> sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target
> vocabulary of about 37000 tokens.
