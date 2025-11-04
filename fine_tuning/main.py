from .config import settings
from .tuning_script import FineTuning
from .helper import train_dataset as helper_train_dataset, test_dataset as helper_test_dataset

def main():

    
    fine_tuner = FineTuning(
        model_name=settings.model_name,
        output_dir=settings.output_dir

    )
    fine_tuner.login_huggingface(settings.huggingface_token)
    # lets load the model and tokenizer
    fine_tuner.load_model_and_tokenize()

    # load the datasets
    train_dataset = helper_train_dataset
    valid_dataset = helper_test_dataset
    fine_tuner.train_dataset = train_dataset
    fine_tuner.valid_dataset = valid_dataset

    # preprocss the datasets
    fine_tuner.preprocessing_dataset()
    # apply the lora
    fine_tuner.apply_lora()
    latest_checkpoint=fine_tuner.find_checkpoint()
    # setup the trainer
    fine_tuner.fine_tune(
        epochs=3,
        batch_size=16,
        lr=1e-4,
        grad_accum=2,
        resume_from_checkpoint=latest_checkpoint
    )

if __name__ == "__main__":
    main()
