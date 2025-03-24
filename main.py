from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

Stage_Name = "Prepare base Model"

try:
    logger.info(f"*********************")
    logger.info(f">>>>>> stage{Stage_Name} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage{Stage_Name} completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e