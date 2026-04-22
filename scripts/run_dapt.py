import torch
from dataclasses import dataclass
from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    set_seed,
)
from datasets import load_dataset

@dataclass
class DataCollatorForSpanLanguageModeling:
    tokenizer: PreTrainedTokenizer
    mlm_probability: float = 0.20

    def __call__(self, examples):
        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        inputs, labels = self.mask_tokens(input_ids)
        return {"input_ids": inputs, "labels": labels}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(v, already_has_special_tokens=True)
            for v in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Merge adjacent masks into 2-token spans
        for i, line in enumerate(masked_indices):
            record_index = -1
            for j, value in enumerate(line):
                if value:
                    if record_index == -1:
                        if j != 0 and masked_indices[i][j - 1]:
                            continue
                        record_index = j
                    else:
                        if record_index == j - 1:
                            record_index = -1
                            continue
                        masked_indices[i][record_index] = False
                        masked_indices[i][j - 1] = True
                        record_index = -1

        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def main():
    set_seed(42)

    DRY_RUN = False  # when True it does a small test to be sure there are no syntax/logic errors, FALSE = real run

    model_name = "bert-base-cased"
    train_file = "data/raw/unlabeled/politics/politics_domainlevel.txt"
    output_dir = "results/models/bert-dapt-politics"
    block_size = 256
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    raw = load_dataset("text", data_files={"train": train_file})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], return_special_tokens_mask=False)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concat["input_ids"]) // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concat.items()
        }

    lm_dataset = tokenized.map(group_texts, batched=True)

    split = lm_dataset["train"].train_test_split(test_size=0.005, seed=42)

    collator = DataCollatorForSpanLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.20
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3 if not DRY_RUN else 1,
        max_steps=10000 if not DRY_RUN else 20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        save_steps=2000,
        save_total_limit=2,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=5000,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DAPT checkpoint saved to {output_dir}]")


if __name__ == "__main__":
    main()