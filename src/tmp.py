from fairseq.models.transformer_lm import TransformerLanguageModel

lg = "it"

model = TransformerLanguageModel.from_pretrained(
    f"models/word/{lg}/transformer",
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=f"models/word/{lg}/transformer/bin/",
)
model.eval()
