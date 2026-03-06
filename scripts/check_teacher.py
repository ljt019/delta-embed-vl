import httpx
from PIL import Image

from delta_embed_vl.data.media import coerce_image_to_rgb, image_to_data_uri
from delta_embed_vl.model.embedding_inputs import EmbeddingInput
from delta_embed_vl.settings import Settings

settings = Settings()


def _build_payload(sample: EmbeddingInput) -> dict[str, object]:
    """Build a vLLM chat-embedding request body for Qwen3-VL."""
    user_content: list[dict[str, object]] = []
    resolved_image = coerce_image_to_rgb(sample.image)
    if sample.image is not None and resolved_image is None:
        raise ValueError("Could not resolve image for teacher embedding request.")

    if resolved_image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(resolved_image)},
            }
        )

    user_content.append({"type": "text", "text": sample.text or ""})

    return {
        "model": settings.teacher_model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sample.instruction or DEFAULT_EMBED_INSTRUCTION,
                    }
                ],
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
    }


def get_embedding(
    text: str | None = None,
    image: Image.Image | None = None,
) -> list[float]:
    """Get a single teacher embedding. Useful for testing / ad-hoc queries."""
    payload = _build_payload(EmbeddingInput(text=text, image=image))
    resp = httpx.post(
        f"{settings.teacher_base_url}/embeddings",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


embedding = get_embedding(text="I love dogs")
print(f"dim={len(embedding)}")
print(embedding[:10])
