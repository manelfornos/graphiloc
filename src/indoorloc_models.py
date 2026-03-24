import torch
import torch.nn as nn
from torch.nn import (
    Dropout,
    LeakyReLU,
    L1Loss,
    CrossEntropyLoss
)
from torch_geometric.nn import (
    Sequential, 
    GraphNorm,
    SAGEConv,
    MLP 
)
import torch_geometric.data as data
import torch.optim as optim


class SAGERegressor(nn.Module):
    """
    GraphSAGE model for regression tasks.
    """
    def __init__(
        self, 
        input_dim: int, 
        gnn_hidden_dims: list, 
        gnn_dropouts: list, 
        mlp_layers: int,
        output_dim: int,
        learning_rate: float,
        lr_factor: float,
        weight_decay: float
    ) -> None:
        super().__init__()
        
        model_layers = []
        gnn_layers = len(gnn_hidden_dims)

        current_dim = input_dim
        for i in range(gnn_layers):
            model_layers.append((SAGEConv(current_dim, gnn_hidden_dims[i], aggr="mean"),
                            "x, edge_index -> x"))
            model_layers.append(GraphNorm(gnn_hidden_dims[i]))  
            model_layers.append(LeakyReLU())

            if i < gnn_layers - 1:  
                model_layers.append(Dropout(p=gnn_dropouts[i]))
            current_dim = gnn_hidden_dims[i]
        
        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        model_layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", model_layers) 

        self.criterion = L1Loss()
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=lr_factor, patience=10
        )

    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data: data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index)
    

class SAGEClassifier(nn.Module):
    """
    GraphSAGE model for classification tasks.
    """
    def __init__(
        self, 
        input_dim: int, 
        gnn_hidden_dims: list, 
        gnn_dropouts: list, 
        mlp_layers: int,
        output_dim: int,
        learning_rate: float,
        lr_factor: float,
        weight_decay: float
    ) -> None:
        super().__init__()
        
        model_layers = []
        gnn_layers = len(gnn_hidden_dims)

        current_dim = input_dim
        for i in range(gnn_layers):
            model_layers.append((SAGEConv(current_dim, gnn_hidden_dims[i], aggr="mean"),
                            "x, edge_index -> x"))
            
            model_layers.append(GraphNorm(gnn_hidden_dims[i]))  
            model_layers.append(LeakyReLU())

            if i < n_layers - 1:  
                model_layers.append(Dropout(p=gnn_dropouts[i]))
            current_dim = gnn_hidden_dims[i]
        
        mlp_layers_dims = [current_dim] + [current_dim // (2 ** (i + 1)) \
                                        for i in range(mlp_layers)] + [output_dim]

        model_layers.append(MLP(mlp_layers_dims))

        self.layers = Sequential("x, edge_index", model_layers) 

        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=lr_factor, patience=10
        )

    def get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data: data.Data) -> torch.Tensor:
        return self.layers(data.x, data.edge_index)
