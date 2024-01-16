from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, Tuple

class MatryoshkaConstrativeLoss(nn.Module):
	"""
	MatryoshkaConstrativeLoss is a custom loss function for contrastive learning with Matryoshka Rrepresentation, which calculates loss with nesting dims.

	Args:
		relative_importance (List[float]): A list of relative importance weights for each matryoshaka dimension term.
		temperature (float): The temperature parameter used in the contrastive loss calculation.
		criterion (nn.Module): The loss criterion used for calculating the similarity matrix.
		device (str): The device on which the loss calculation should be performed.
		**kwargs: Additional keyword arguments to be passed to the criterion.
	"""

	def __init__(self, relative_importance: List[float] = None, temperature: float = None,
				 criterion=nn.CrossEntropyLoss, device: str = None, **kwargs):
		super().__init__()
		self.criterion = criterion(**kwargs)
		self.relative_importance = relative_importance
		self.temperature = temperature
		self.device = device

	def _contrastive_loss(self, emb1, emb2):
		"""
		Calculates the contrastive loss between two sets of embeddings.

		Args:
			emb1 (Tuple[torch.Tensor]): The first set of embeddings.
			emb2 (Tuple[torch.Tensor]): The second set of embeddings.

		Returns:
			List[torch.Tensor]: A list of contrastive loss values for each matryoshka dimension of embeddings.

		"""
		losses = []
		labels = torch.arange(emb1[0].shape[0]).long().to(self.device)
		for emb1_matry, emb2_matry in zip(emb1, emb2):
			sim_matrix_matry = F.cosine_similarity(emb1_matry.unsqueeze(1), emb2_matry.unsqueeze(0), dim=-1) / self.temperature
			loss = self.criterion(sim_matrix_matry, labels)
			losses.append(loss)

		return losses

	def forward(self, emb1: Tuple[torch.Tensor], emb2: Tuple[torch.Tensor]) -> torch.Tensor:
		losses = self._contrastive_loss(emb1, emb2)
		losses = torch.stack(losses)
		rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)

		weighted_losses = rel_importance * losses
		return weighted_losses.sum()


class MatryoshkaLinearLayer(nn.Module):
	"""
	A PyTorch module for Matryoshka Linear Layer.

	Args:
		input_dim (int): The input dimension of the layer. Default is 768.
		out_dim (int): The output dimension of the layer. If None, it is set to input_dim. Default is None.
		nesting_dims (List[int]): The dimensions of the nesting matryoshka dimension. If None, it is calculated based on the input_dim. Default is None.
		efficient (bool): Whether to use efficient Matryoshka projection. Default is True.
		**kwargs: Additional keyword arguments to be passed to the nn.Linear module.

	Attributes:
		input_dim (int): The input dimension of the layer.
		out_dim (int): The output dimension of the layer.
		efficient_matryoshka (bool): Whether to use efficient Matryoshka projection.
		nesting_dims (List[int]): The dimensions of the nested matryoshka representation.

	"""

	def __init__(self, 
				 input_dim: int = 768,
				 out_dim: int = None,
				 nesting_dims: List[int] = None, 		  
				 efficient: bool = True, 
				 **kwargs):
		super().__init__()
		self.input_dim = input_dim
		self.out_dim = input_dim if out_dim is None else out_dim
		self.efficient_matryoshka = efficient
		self.nesting_dims = [2**i for i in range(7, int(math.log2(self.input_dim))+1)] if nesting_dims is None else nesting_dims

		if self.efficient_matryoshka:
			setattr(self, f"nesting_projection_{0}", nn.Linear(self.input_dim, self.out_dim, **kwargs))
		else:
			for i, nested_dim in enumerate(self.nesting_dims):
				setattr(self, f"nesting_projection_{i}", nn.Linear(nested_dim, self.out_dim, **kwargs))


	def forward(self, x):
		"""
		Forward pass of the MatryoshkaLinearLayer.

		Args:
			x (torch.Tensor): The input tensor.

		Returns:
			torch.Tensor: The output tensor containing the nesting logits.

		"""
		nesting_logits = ()
		for i, nested_dim in enumerate(self.nesting_dims):
			if self.efficient_matryoshka:
				projection: nn.Linear = getattr(self, f'nesting_projection_{0}')
				if projection.bias is None:
					nesting_logits += (torch.matmul(x[:, :nested_dim], (projection.weight[:, :nested_dim]).t()), )
				else:
					nesting_logits += (torch.matmul(x[:, :nested_dim], (projection.weight[:, :nested_dim]).t()) + projection.bias, )
			else: 
				projection: nn.Linear = getattr(self, f'nesting_projection_{i}')
				nesting_logits += (projection(x[:, :nested_dim]))

		return nesting_logits


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following. 
    It effectively just look at the first "in_features" for the classification. 
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
        
