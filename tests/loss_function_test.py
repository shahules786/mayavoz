import pytest
import torch

from enhancer.loss import mean_absolute_error, mean_squared_error

loss_functions = [mean_absolute_error(), mean_squared_error()]


def check_loss_shapes_compatibility(loss_fun):

    batch_size = 4
    shape = (1, 1000)
    loss_fun(torch.rand(batch_size, *shape), torch.rand(batch_size, *shape))

    with pytest.raises(TypeError):
        loss_fun(torch.rand(4, *shape), torch.rand(6, *shape))


@pytest.mark.parametrize("loss", loss_functions)
def test_loss_input_shapes(loss):
    check_loss_shapes_compatibility(loss)


@pytest.mark.parametrize("loss", loss_functions)
def test_loss_output_type(loss):

    batch_size = 4
    prediction, target = torch.rand(batch_size, 1, 1000), torch.rand(
        batch_size, 1, 1000
    )
    loss_value = loss(prediction, target)
    assert isinstance(loss_value.item(), float)
