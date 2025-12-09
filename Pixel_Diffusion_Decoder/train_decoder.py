from trainer.cascade512_trainer import Cascade512Trainer

train_index = "your_path/train/index.jsonl"
val_index   = "your_path/val/index.jsonl"

trainer = Cascade512Trainer(train_index, val_index, bs=4, lr=2e-5)
trainer.train(epochs=30, patience=10, vis_steps_stage2=200)
