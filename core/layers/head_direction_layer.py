import torch


class HeadDirectionLayer:
    def __init__(
        self,
        num_cells: int,
        theta_0: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Discretized head-direction layer."""
        self.num_cells = num_cells
        self.theta_0 = theta_0
        self.dtype = dtype
        self.device = device

        theta_i = torch.linspace(
            start=0,
            end=2 * torch.pi,
            steps=self.num_cells + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        self.tuning_kernel = torch.stack(
            [
                torch.cos(theta_i + self.theta_0),
                torch.sin(theta_i + self.theta_0),
            ],
            dim=1,
        ).to(dtype=dtype, device=device)

        self.hd_activations = None

    def get_hd_activation(self, v_in) -> torch.Tensor:
        """Compute head-direction activations from a 2D heading vector."""
        if isinstance(v_in, torch.Tensor):
            v_in = v_in.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            v_in = torch.tensor(data=v_in, dtype=self.dtype, device=self.device)

        if v_in.shape != (2,):
            raise ValueError("v_in should be a 2-element vector.")

        activation = torch.matmul(self.tuning_kernel, v_in)
        self.hd_activations = activation
        return activation
