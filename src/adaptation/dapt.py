from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments

def run_dapt(model_name, dataset, output_dir):
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=5000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    return model