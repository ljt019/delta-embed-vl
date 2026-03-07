import argparse

from PIL import Image

from delta_embed_vl.model.embedding_inputs import EmbeddingInput
from delta_embed_vl.model.teacher import load_teacher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test the local in-process teacher embedder."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text", default="I love dogs")
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    image = None
    if args.image_size is not None:
        image = Image.new("RGB", (args.image_size, args.image_size), (255, 255, 255))

    teacher = load_teacher(device=args.device)
    embedding = teacher.embed([EmbeddingInput(text=args.text, image=image)])[0]
    print(f"dim={embedding.shape[0]}")
    print(embedding[:10].tolist())


if __name__ == "__main__":
    main()
