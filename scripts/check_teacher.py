from delta_embed_vl.teacher.generate import get_embedding

embedding = get_embedding(text="I love dogs")
print(f"dim={len(embedding)}")
print(embedding[:10])
