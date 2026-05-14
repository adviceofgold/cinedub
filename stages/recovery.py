from ..core import Config, PipelineContext, Stage, StageResult, RuntimeMode
from ..checkpoint import CheckpointManager
from ..logging import CineDubLogger


class RecoveryStage(Stage):
    def __init__(self):
        super().__init__("recovery")

    def execute(self, ctx: PipelineContext) -> StageResult:
        log = CineDubLogger.get()
        checkpoint = CheckpointManager(ctx.config)

        completed = checkpoint.completed_stages()
        if not completed:
            log.info("No checkpoints found, starting from beginning")
            return StageResult(True, self.name,
                               message="No checkpoints to recover from")

        log.info(f"Found {len(completed)} completed stages")

        latest_data = checkpoint.load(completed[-1])
        if latest_data:
            log.info(f"Latest completed: {completed[-1]} "
                     f"(at {latest_data.get('timestamp', 'unknown')})")

        return StageResult(True, self.name,
                           message=f"Found {len(completed)} completed stages",
                           metrics={"completed_stages": completed})
