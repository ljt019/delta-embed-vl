from delta_embed_vl import cfg, configure_logging, resolve_attention, set_seed
from delta_embed_vl.evals.mteb_eval import run_eval as run_mteb_eval


def eval_model() -> None:
    model_path = f"checkpoints/{cfg['name']}"
    run_mteb_eval(
        model_path=model_path,
        eval_batch_size=cfg["train"]["batch_size"],
        max_length=cfg["max_length"],
        attention=resolve_attention(cfg["attention"]),
    )


def eval_model_cli() -> None:
    configure_logging()
    set_seed(cfg["train"]["seed"])
    eval_model()


if __name__ == "__main__":
    eval_model_cli()
