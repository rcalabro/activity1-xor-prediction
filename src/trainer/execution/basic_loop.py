from .base import ExecutionStrategy

class BasicLoopExecution(ExecutionStrategy):
    def __init__(self, **options):
        super().__init__(**options)
        self.name = 'basic-loop'

    def _execute(self, nn, X, y, training_strategy, **options):
        epochs = self.epochs
        history = []
        success = False

        for epoch in range(epochs):
            loss = training_strategy.train_step(nn, X, y, **options)
            history.append(loss)

            if loss <= self.target_error:
                success = True
                break

            if epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1:
                self._log_epoch(epoch, loss)

        return history, epoch + 1, success
