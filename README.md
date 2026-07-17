# Fashion Multi-Expert Qdrant Indexer

This project indexes local fashion images into one Qdrant point per image using three independent named vector spaces:

| Named vector | Source branch | Size | Distance |
|---|---|---:|---|
| `garment_vector` | Grounding DINO crop → Fashion-CLIP projection | 512 | Cosine |
| `region_vector` | Full-image Google ViT final CLS state | 768 | Cosine |
| `vibe_vector` | Full-image LAION DataComp OpenCLIP projection | 768 | Cosine |

The collection is named `fashion_multi_expert_search` and is persisted locally at `./qdrant_fashion_db` by default. Every point has a UUID v4 identifier and exactly this payload:

```json
{
  "image_path": "absolute/path/to/image.jpg",
  "file_hash": "raw-byte-md5",
  "timestamp": "ISO-8601 UTC timestamp"
}
```

## Install

Python 3.10 or newer is required. A CUDA GPU is strongly recommended because the four loaded model components are large.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

## Index images

Place images under `images/`, including nested folders if desired, then run:

```bash
python index_fashion.py
```

Equivalent installed command:

```bash
fashion-index --images-dir images --db-path qdrant_fashion_db
```

Useful options:

```text
--device {auto,cuda,mps,cpu}
--precision {auto,float32,float16,bfloat16}
--batch-size 8
--box-threshold 0.25
--text-threshold 0.25
--max-image-pixels 100000000
--max-image-dimension 2048
--no-recursive
--no-skip-existing
--recreate-collection
```

`--recreate-collection` deletes the configured collection before rebuilding it. Without that explicit flag, an existing incompatible schema produces a clear error and is never overwritten.

The first run downloads model weights from Hugging Face. The loader selects CUDA first, then Apple MPS, then CPU. `auto` precision uses float16 on CUDA and float32 elsewhere; use `--precision float32` if a GPU operator is unstable in half precision.

## Input safety and repeat runs

- Files are discovered deterministically and filtered by supported suffix.
- MD5 is computed over the original raw bytes in chunks. Byte-identical files are decoded and indexed only once.
- PIL verifies each file, rejects decompression bombs and oversized images, applies EXIF orientation, composites transparency onto white, converts to RGB, and bounds the longest edge before model-specific preprocessing.
- Each model's official processor performs its required resize and pixel normalization. A shared normalized tensor is intentionally not reused because the checkpoints use different preprocessing statistics.
- Existing `file_hash` payloads are loaded before inference, so repeated runs do not add new UUID points for already indexed images.
- A point is written only after all three finite, correctly sized, L2-normalized embeddings succeed.
- Every Hugging Face model is pinned to a full commit revision. A JSON index manifest beside the Qdrant directory records those revisions, detection thresholds, global preprocessing, compute precision, and embedding-library versions. Appending vectors is refused if that contract differs, preventing mixed embedding spaces.

The three expert branches are logically independent. They are evaluated sequentially for each image on the selected accelerator to bound peak activation memory and avoid unsafe concurrent access to one CUDA/MPS device; each result remains an independent named index rather than being fused into one vector.

## Model identifier note

The requested vibe checkpoint is used exactly: [`laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K). Its upstream model card and OpenCLIP configuration identify it as a CLIP ViT-L/14 trained on DataComp-1B with a 768-dimensional projection. It is not an EVA02 architecture, despite sometimes being described that way. The implementation labels the branch by its purpose (`VibeAestheticEncoder`) and does not misidentify the checkpoint architecture.

Other authoritative model/API references:

- [Grounding DINO Transformers documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino)
- [Fashion-CLIP model card](https://huggingface.co/patrickjohncyh/fashion-clip)
- [Google ViT model card](https://huggingface.co/google/vit-base-patch16-224)
- [Qdrant named-vector collection documentation](https://qdrant.tech/documentation/manage-data/collections/)

## Tests

The test suite exercises image validation/deduplication, crop geometry, normalization contracts, Qdrant named-vector schema enforcement, UUID payload creation, and a model-free pipeline run. It does not download model weights.

```bash
python -m pip install -e ".[dev]"
pytest
```
