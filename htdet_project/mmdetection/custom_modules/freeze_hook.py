"""
FreezeBackboneHook: freezes all model params except FA/CBAM attention modules
for the first `freeze_epochs` epochs, then unfreezes everything.

Usage in config:
    custom_hooks = [
        dict(type='NumClassCheckHook'),
        dict(type='FreezeBackboneHook', freeze_epochs=10),
    ]
"""
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeBackboneHook(Hook):

    def __init__(self, freeze_epochs=10):
        self.freeze_epochs = freeze_epochs
        self._frozen = False

    def _attention_param(self, name):
        return any(k in name for k in ('fa_modules', 'cbam_modules'))

    def before_run(self, runner):
        self._freeze(runner)

    def before_epoch(self, runner):
        if runner.epoch == self.freeze_epochs and self._frozen:
            self._unfreeze(runner)

    def _freeze(self, runner):
        model = runner.model
        frozen, total = 0, 0
        for name, p in model.named_parameters():
            total += 1
            if not self._attention_param(name):
                p.requires_grad_(False)
                frozen += 1
        runner.logger.info(
            f'FreezeBackboneHook: froze {frozen}/{total} params. '
            f'Only FA/CBAM modules will train for {self.freeze_epochs} epochs.')
        self._frozen = True

    def _unfreeze(self, runner):
        model = runner.model
        for p in model.parameters():
            p.requires_grad_(True)
        runner.logger.info(
            f'FreezeBackboneHook: unfroze all params at epoch {runner.epoch}. '
            f'Full model fine-tuning begins.')
        self._frozen = False
