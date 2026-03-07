import argparse

from PIL import Image

from delta_embed_vl.model.embedding_inputs import EmbeddingInput
from delta_embed_vl.model.student import embed, get_embedding_dim, load_student


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test the local student embedder."
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--text", default="I love dogs")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    image = None
    if args.image_size is not None:
        image = Image.new("RGB", (args.image_size, args.image_size), (255, 255, 255))

    model, processor, projection_head = load_student(device=args.device)
    embeddings = embed(
        model,
        projection_head,
        processor,
        [EmbeddingInput(text=args.text, image=image)],
        max_length=args.max_length,
    )
    embedding = embeddings[0].detach().cpu()

    print(f"dim={get_embedding_dim(model, projection_head)}")
    print(f"norm={embedding.norm().item():.6f}")
    print(embedding[:10].tolist())


if __name__ == "__main__":
    main()
