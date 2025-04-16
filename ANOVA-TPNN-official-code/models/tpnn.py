import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit
from .utils import check_numpy
from warnings import warn



class TPNN(ModuleWithInit):
    def __init__(self, in_features, num_tpnn, output_dim=1, 
                 bin_function=sparsemoid,
                 initialize_response_=nn.init.normal_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0, device= "cpu" ,monotone=False):
        super().__init__()
        
        
        self.device = device
        self.in_features, self.num_tpnn, self.output_dim = in_features, num_tpnn, output_dim
        self.bin_function = bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.monotone = monotone
        self.relu = nn.ReLU()
                
        
        self.response_ = nn.Parameter(torch.zeros([output_dim,num_tpnn]), requires_grad=True)
        
        initialize_response_(self.response_)
        
        self.final_response = nn.Parameter(torch.stack([self.response_]*(2**self.in_features),dim=2), requires_grad=False)
        

        self.feature_thresholds = nn.Parameter(
            torch.full([num_tpnn, in_features], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_tpnn, in_features], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.in_features)
            offsets = 2 ** torch.arange(self.in_features)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [in_features, 2 ** in_features, 2]

    def forward(self, input,training=True):
        #defalut
        self.training = training
        
        feature_selectors  = torch.zeros([self.in_features,self.num_tpnn,self.in_features]).to(self.device)
        for i in range(self.in_features):
            feature_selectors[i][:,i] = 1
      

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_tpnn, in_features]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_tpnn, in_features, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_tpnn, in_features, 2], approximately binary
        
        if self.training == True:
            # For satisfying sum-to-zero condtion
            const1 = bins.sum(axis=0)[:,:,1]
            const2 = bins.sum(axis=0)[:,:,0]
            ratio = - const2 /const1
            ones = torch.ones(ratio.shape).to(input.device)
            
            ratio = ratio.unsqueeze(-1)
            ones = ones.unsqueeze(-1)
            id_const = torch.concat([ones,ratio],axis=-1)
            self.id_const = id_const
            
        else:
            id_const = self.final_id_const
            
        bins = bins * id_const
        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
            
        weight = torch.prod(bin_matches, dim=-2)
        
        if self.in_features == 1:
            if self.monotone == "incre":
                output = ( weight.sum(axis=-1) * (-1)*self.relu(self.response_) ).reshape(input.shape[0],self.num_tpnn,self.output_dim)
            elif self.monotone == "decre":
                output = ( weight.sum(axis=-1) * self.relu(self.response_) ).reshape(input.shape[0],self.num_tpnn,self.output_dim)
            else:
                output = ( weight.sum(axis=-1) * self.response_ ).reshape(input.shape[0],self.num_tpnn,self.output_dim)
        else:
            output = ( weight.sum(axis=-1) * self.response_ ).reshape(input.shape[0],self.num_tpnn,self.output_dim)

   
        return output

    #Save the constant for sum-to-zero condition.
    def save_id_constants(self):
        self.final_id_const = nn.Parameter( self.id_const, requires_grad=False )
            

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():

            feature_selectors  = torch.zeros([self.in_features,self.num_tpnn,self.in_features]).to(self.device)
            for i in range(self.in_features):
                feature_selectors[i][:,i] = 1    
            
            # ^--[in_features, num_tpnn, in_features]

            feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            # ^--[batch_size, num_tpnn, in_features]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_tpnn, self.in_features])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_tpnn, self.in_features)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return "{}(in_features={}, num_tpnn={}, in_features={}, output_dim={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_tpnn, self.in_features, self.output_dim
        )
        