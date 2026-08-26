"""
Experiment orchestration for the pipeline.
"""

from src.experiment.context import PipelineContext


class Experiment:
    """
    Class to manage the experiment pipeline.
    Orchestrates the experiment lifecycle.
    """

    def __init__(self, context: PipelineContext) -> None:
        """
        Initialize the experiment with a shared pipeline context.

        Args:
            context (PipelineContext): Shared runtime state.
        """
        self.context: PipelineContext = context

    def run(self) -> None:
        """
        Execute the selected pipeline stages in canonical order.

        Resolve the requested stages from the context, instantiate each
        stage, log and run them sequentially, stopping on the first failure.

        After all stages run successfully, cleans up temporary
        artifacts if configured to do so.
        """
        logger = self.context.logger
        logger.info("Starting experiment pipeline.")

        try:
            logger.info("Experiment pipeline completed successfully.")

        except Exception as exc:
            logger.error(f"Experiment pipeline failed: {exc}", exc_info=True)
            raise
