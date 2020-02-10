import pytest

from abc import ABC
from collections import OrderedDict
import sys
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear
from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer

from neuralmagicML.recal import Modifier, ScheduledModifier, ScheduledUpdateModifier


def def_model():
    return Sequential(
        OrderedDict(
            [
                ("fc1", Linear(8, 16, bias=True)),
                ("fc2", Linear(16, 32, bias=True)),
                (
                    "block1",
                    Sequential(
                        OrderedDict(
                            [
                                ("fc1", Linear(32, 16, bias=True)),
                                ("fc2", Linear(16, 8, bias=True)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )


def def_optim_sgd(model):
    return SGD(model.parameters(), lr=0.0001)


def def_optim_adam(model: Module):
    return Adam(model.parameters())


class ModifierTest(ABC):
    def create_test_objs(self, modifier_lambda, model_lambda, optim_lambda):
        modifier = modifier_lambda()
        model = model_lambda()
        optim = optim_lambda(model)

        return modifier, model, optim

    def initialize_helper(
        self, modifier: Modifier, model: Module, optimizer: Optimizer
    ):
        modifier.initialize(model, optimizer)

    @pytest.mark.device_cpu
    def test_initialize(self, modifier_lambda, model_lambda, optim_lambda):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        self.initialize_helper(modifier, model, optimizer)
        assert modifier.initialized

    @pytest.mark.device_cpu
    def test_update(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_epoch,
        test_steps_per_epoch,
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

    @pytest.mark.device_cpu
    def test_loss_update(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_loss,
        test_epoch,
        test_steps_per_epoch,
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.loss_update(
                test_loss, model, optimizer, test_epoch, test_steps_per_epoch
            )

        self.initialize_helper(modifier, model, optimizer)
        new_loss = modifier.loss_update(
            test_loss, model, optimizer, test_epoch, test_steps_per_epoch
        )

        assert isinstance(new_loss, Tensor)

    @pytest.mark.device_cpu
    def test_optimizer_pre_step(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_epoch,
        test_steps_per_epoch,
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        modifier.optimizer_pre_step(model, optimizer, test_epoch, test_steps_per_epoch)

    @pytest.mark.device_cpu
    def test_optimizer_post_step(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_epoch,
        test_steps_per_epoch,
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        modifier.optimizer_pre_step(model, optimizer, test_epoch, test_steps_per_epoch)


class ScheduledModifierTest(ModifierTest):
    def start_helper(self, modifier: Modifier, model: Module, optimizer: Optimizer):
        modifier._started = True

    @pytest.mark.device_cpu
    def test_start_pending(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.start_pending(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        modifier.enabled = False
        assert not modifier.start_pending(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True

        if modifier.start_epoch < 0.0:
            assert modifier.start_pending(0.0, test_steps_per_epoch)
        elif modifier.start_epoch > 0.0:
            assert not modifier.start_pending(0.0, test_steps_per_epoch)
            assert modifier.start_pending(modifier.start_epoch, test_steps_per_epoch)

    @pytest.mark.device_cpu
    def test_end_pending(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.start_pending(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        self.start_helper(modifier, model, optimizer)
        modifier.enabled = False
        assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True

        if modifier.end_epoch < 0.0:
            assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
        elif modifier.end_epoch > 0.0:
            assert not modifier.end_pending(0.0, test_steps_per_epoch)
            assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
            assert modifier.end_pending(modifier.end_epoch, test_steps_per_epoch)

    @pytest.mark.device_cpu
    def test_update_ready(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.update_ready(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)
        modifier.enabled = False
        assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True
        assert modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)

        self.start_helper(modifier, model, optimizer)

        if modifier.end_epoch < 0.0:
            assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        elif modifier.end_epoch > 0.0:
            assert not modifier.update_ready(0.0, test_steps_per_epoch)
            assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
            assert modifier.update_ready(modifier.end_epoch, test_steps_per_epoch)

    @pytest.mark.device_cpu
    def test_scheduled_update(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )

        with pytest.raises(RuntimeError):
            modifier.start_pending(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model, optimizer)

        if modifier.start_epoch <= 0.0:
            modifier.scheduled_update(model, optimizer, 0.0, test_steps_per_epoch)
        else:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(model, optimizer, 0.0, test_steps_per_epoch)

            modifier.scheduled_update(
                model, optimizer, modifier.start_epoch, test_steps_per_epoch
            )

        self.start_helper(modifier, model, optimizer)

        if modifier.end_epoch < 0.0:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(
                    model, optimizer, modifier.start_epoch, test_steps_per_epoch
                )
        elif modifier.end_epoch > 0.0:
            modifier.scheduled_update(
                model, optimizer, modifier.end_epoch, test_steps_per_epoch
            )

    @pytest.mark.device_cpu
    def test_update(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_epoch,
        test_steps_per_epoch,
    ):
        with pytest.raises(RuntimeError):
            super().test_update(
                modifier_lambda,
                model_lambda,
                optim_lambda,
                test_epoch,
                test_steps_per_epoch,
            )


class ScheduledUpdateModifierTest(ScheduledModifierTest):
    def start_helper(
        self, modifier: ScheduledUpdateModifier, model: Module, optimizer: Optimizer
    ):
        super().start_helper(modifier, model, optimizer)
        modifier._last_update_epoch = modifier.start_epoch

    @pytest.mark.device_cpu
    def test_update_ready(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        super().test_update_ready(
            modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
        )
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        self.start_helper(modifier, model, optimizer)
        min_update_freq = 1.0 / float(test_steps_per_epoch)

        if modifier.update_frequency <= min_update_freq + sys.float_info.epsilon:
            assert modifier.update_ready(
                modifier.start_epoch + min_update_freq, test_steps_per_epoch
            )
        else:
            assert not modifier.update_ready(
                modifier.start_epoch + min_update_freq, test_steps_per_epoch
            )
            assert modifier.update_ready(
                modifier.start_epoch + modifier.update_frequency, test_steps_per_epoch
            )

    @pytest.mark.device_cpu
    def test_scheduled_update(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        super().test_scheduled_update(
            modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
        )
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
        self.initialize_helper(modifier, model, optimizer)
        self.start_helper(modifier, model, optimizer)
        min_update_freq = 1.0 / float(test_steps_per_epoch)

        if modifier.update_frequency <= min_update_freq + sys.float_info.epsilon:
            modifier.scheduled_update(
                model,
                optimizer,
                modifier.start_epoch + min_update_freq,
                test_steps_per_epoch,
            )
        else:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(
                    model,
                    optimizer,
                    modifier.start_epoch + min_update_freq,
                    test_steps_per_epoch,
                )

            modifier.scheduled_update(
                model,
                optimizer,
                modifier.start_epoch + modifier.update_frequency,
                test_steps_per_epoch,
            )


@pytest.fixture
def test_epoch():
    return 0.0


@pytest.fixture
def test_steps_per_epoch():
    return 100


@pytest.fixture
def test_loss():
    return torch.tensor(0.0)


@pytest.mark.parametrize("modifier_lambda", [lambda: Modifier()], scope="function")
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [def_optim_sgd, def_optim_adam], scope="function"
)
class TestModifierImpl(ModifierTest):
    pass


@pytest.mark.parametrize(
    "modifier_lambda", [lambda: ScheduledModifier()], scope="function"
)
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [def_optim_sgd, def_optim_adam], scope="function"
)
class TestScheduledModifierImpl(ScheduledModifierTest):
    pass


@pytest.mark.parametrize(
    "modifier_lambda", [lambda: ScheduledUpdateModifier()], scope="function"
)
@pytest.mark.parametrize("model_lambda", [def_model], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [def_optim_sgd, def_optim_adam], scope="function"
)
class TestScheduledUpdateModifierImpl(ScheduledUpdateModifierTest):
    pass
