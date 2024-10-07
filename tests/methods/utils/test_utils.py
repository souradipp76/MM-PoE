import os
import random
import numpy as np
import torch
import pytest
from unittest import mock

from mm_poe.methods.utils.utils import set_seed, load_data

@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_set_seed(seed):
    # Mocking torch.cuda methods to avoid actual CUDA calls during the test
    with mock.patch("torch.cuda.manual_seed_all") as mock_cuda_seed_all:

        # Call the function with the seed
        set_seed(seed)

        # Check if os environment variable is set correctly
        assert os.environ['PYTHONHASHSEED'] == str(seed)

        # Check if torch manual_seed is called correctly
        assert torch.initial_seed() == seed

        # Check if CUDA seeds were set correctly (mocked)
        mock_cuda_seed_all.assert_called_with(seed)