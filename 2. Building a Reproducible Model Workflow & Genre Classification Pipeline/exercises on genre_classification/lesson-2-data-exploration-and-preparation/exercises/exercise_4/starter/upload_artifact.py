import logging
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go():

    logger.info("Creating run")
    with wandb.init(project= "exercise_4", 
                    job_type= "upload_file") as run:
        artifact = wandb.Artifact(
                                name = 'exercise_4/genres_mod.parquet',
                                type = 'raw_data',
                                description = '"A modified version of the songs dataset"'
        )

        logger.info("Creating artifact")
        artifact.add_file('./genres_mod.parquet')

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        run.finish()



if __name__ == "__main__":
    go()