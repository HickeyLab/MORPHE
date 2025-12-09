from trainers.Latent_DIffusion_Trainer import LatentTrainer

if __name__ == "__main__":
    trainer = LatentTrainer()
    trainer.train(epochs=10)
