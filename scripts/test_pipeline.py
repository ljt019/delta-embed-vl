import logging

import numpy as np

from delta_embed_vl.data.preprocess import (
    preprocess_cauldron_config,
    preprocess_wikipedia,
)
from delta_embed_vl.teacher.generate import embed_cauldron_config, embed_wikipedia

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LIMIT = 10

logger.info("=== Wikipedia (limit=%d) ===", LIMIT)
wiki_ds = preprocess_wikipedia(limit=LIMIT)
logger.info("Preprocessed: %d samples", len(wiki_ds))
logger.info("Sample: %s", wiki_ds[0]["text"][:100])

wiki_emb = embed_wikipedia(limit=LIMIT)
logger.info("Embeddings shape: %s", wiki_emb.shape)
logger.info("First vector norm: %.4f", np.linalg.norm(wiki_emb[0]))

logger.info("=== Cauldron/vqav2 (limit=%d) ===", LIMIT)
cauldron_ds = preprocess_cauldron_config("vqav2", limit=LIMIT)
logger.info("Preprocessed: %d samples", len(cauldron_ds))
if len(cauldron_ds) > 0:
    logger.info("Sample text: %s", cauldron_ds[0].get("text", "")[:100])
    logger.info("Has image: %s", cauldron_ds[0].get("image") is not None)
else:
    logger.info("No usable vqav2 rows after preprocess normalization")

cauldron_emb = embed_cauldron_config("vqav2", limit=LIMIT)
logger.info("Embeddings shape: %s", cauldron_emb.shape)
if cauldron_emb.shape[0] > 0:
    logger.info("First vector norm: %.4f", np.linalg.norm(cauldron_emb[0]))
else:
    logger.info("No vqav2 embeddings were generated")

logger.info("=== Pipeline test passed ===")
