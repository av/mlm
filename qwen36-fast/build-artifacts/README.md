# Build artefacts (git-ignored GGUFs)

Contents are not tracked in git (see .gitignore). Large files such as the
merged MTP GGUF live here to avoid blowing up the repo.

## qwen36-27b-mtp-merged.gguf

Produced by `../patches/inject_mtp.py`. 11.83 GiB. Derived from the Unsloth
`Qwen3.6-27B-UD-Q2_K_XL.gguf` (HF cache) with 15 MTP tensors from
`/tmp/qwen36-mtp-shards/` appended as layer 64 (F16), plus a new
`qwen35.nextn_predict_layers=1` KV and `block_count=65`.

To regenerate:
```
python3 ../patches/inject_mtp.py
```
