import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpnn import TPNN

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from itertools import combinations

from models import nn_utils
import pickle


from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, random_state=0,output_distribution='uniform')


class ANOVA_Ensemble(nn.Sequential):
    def __init__(self, features_list, 
                 tpnn_list,
                 device, 
                 training=True, 
                 input_dropout=0.0, **kwargs):
        
        self.training = training
        self.features_list = features_list
        self.device = device 

        super().__init__(*tpnn_list)
        
        self.input_dropout = input_dropout
        
        self.tpnn_list = tpnn_list
        
        self.drop_out = False


    def forward(self, x,inital=False):
        #initial_features = x.shape[-1]
        
        count = 0
        output_list = torch.tensor([]).to(self.device)
        
        for tpnn in self.tpnn_list:
            if type(self.features_list[count]) == int:
                tpnn_inp = x[:,self.features_list[count]].reshape(-1,1)
            else:
                tpnn_inp = x[:,self.features_list[count]]
                
            if inital:
                tpnn.initialize(tpnn_inp)
                
            h = tpnn(tpnn_inp,self.training).sum(axis=1)
            
            output_list = torch.cat([output_list,h],axis=1)
            count += 1
        #output_list = output_list[:,self.num_features:]
  
        return output_list
    
    def model_save_id_constants(self):
        for tpnn in self.tpnn_list:
            tpnn.save_id_constants()
                       
    
    
def gen_tpnn(in_features,
             num_Ks,
             device,
             num_multiclass,
             max_order,
             bin_function,
             features_list="all",
             monotone_list=[]):
    
    if features_list =="all" :
        if max_order == 1:
            features_list = [i for i in range(in_features)]
        elif max_order == 2:
            features_list = [i for i in range(in_features)]
            features_list.extend(list( itertools.combinations(features_list,2) ))    
        elif max_order == 3:
            features_list = [i for i in range(10)]
            features_list_copy = copy.deepcopy(features_list)
            features_list.extend(list( itertools.combinations(features_list_copy,2) ))
            features_list.extend(list( itertools.combinations(features_list_copy,3) )) 
        elif max_order == 4:
            features_list = [i for i in range(10)]
            features_list_copy = copy.deepcopy(features_list)
            features_list.extend(list( itertools.combinations(features_list_copy,2) ))
            features_list.extend(list( itertools.combinations(features_list_copy,3) )) 
            features_list.extend(list( itertools.combinations(features_list_copy,4) ))
        else:
            print("Order error")            
        
    if num_multiclass <= 2:
        output_dim = 1
    elif num_multiclass >= 3:
        output_dim = num_multiclass
        
        
    tpnn_list = []
    for i in range(len(features_list)):
                            
        if type(features_list[i]) == int:
            in_features = 1
            if len(monotone_list) != 0:
                if monotone_list[i] != False:
                    monotone = monotone_list[i]
                else:
                    monotone = False
            else:
                monotone = False
        else:
            in_features = len(features_list[i])
            monotone = False

                                         
        tpnn = TPNN(in_features=in_features, 
                    num_tpnn=num_Ks,
                    output_dim = output_dim, 
                    device= device, 
                    bin_function=bin_function,
                    monotone=monotone)

        tpnn_list.append(tpnn) 
        
    return features_list, tpnn_list




class Bases_ANOVA_Ensemble(nn.Sequential):
    def __init__(self, features_list, tpnn_list , num_bases,device, training=True ,input_dropout=0.0, **kwargs):
        
        self.training = training
        self.features_list = features_list
        self.device = device 

        super().__init__(*tpnn_list)
        
        self.input_dropout = input_dropout
        
        self.tpnn_list = tpnn_list
        
        self.drop_out = False
        self.num_bases = num_bases
        self._num_classes = 1

        self.featurizer = nn.Conv1d(
            in_channels=len(self.features_list) * self.num_bases,
            out_channels=len(self.features_list),
            kernel_size=1,
            groups=len(self.features_list),
        )
        
        self.classifier = nn.Linear(
                in_features = len(self.features_list),
                out_features=self._num_classes,
                bias=True,
            )
        
    def forward(self, x,inital=False):
        
        #grid 버전
        #gx,gy = np.meshgrid(np.array(x.reshape(-1,1).cpu().detach()),np.array(x.reshape(-1,1).cpu().detach()))
        #all_x_2 = torch.concat([torch.tensor(gx).reshape(-1,1),torch.tensor(gy).reshape(-1,1)]).to(self.device)
        output_list = torch.tensor([]).to(self.device)

        if False:
            tpnn_list_1 = self.tpnn_list[0]
            
            output_1 = tpnn_list_1.forward(x.reshape(-1,1),self.training,x.reshape(-1,1))
            output_list = output_1.reshape(x.shape[0],x.shape[1],-1)
        
        else:
            for i in range(0,len(self.features_list)):
            
                if type(self.features_list[i]) == int:
                    bases_tpnn =self.tpnn_list[0]
                    tpnn_input = x[:,self.features_list[i]].reshape(-1,1)
                
                else:
                    bases_tpnn = self.tpnn_list[1]
                    tpnn_input = x[:,self.features_list[i]]
                    
                if inital:
                    bases_tpnn.initialize(tpnn_input)        
                        
                h = bases_tpnn.forward(tpnn_input,
                            training = self.training)
                    
                                    
                output_list = torch.cat([output_list,h],axis=1) 
        
         
        out_feats = self.featurizer(output_list.reshape(x.shape[0], -1, 1)).squeeze(-1)
        #out_feats = output_list.sum(axis=2)
        
        out = self.classifier(out_feats)
        
        return out,out_feats
    
    def model_save_id_constants(self):
        for bases_tpnn in self.tpnn_list:
            bases_tpnn.save_id_constants()
    
    
def bases_gen_odst(in_features,
                   device,
                   max_order,
                   bin_function,
                   features_list="all",
                   num_bases=100,
                   num_multiclass = 1,
                   monotone_list=[]):
    

    if features_list =="all" :
        if max_order == 1:
            features_list = [i for i in range(in_features)]
        elif max_order == 2:
            features_list = [i for i in range(in_features)]
            features_list.extend(list( itertools.combinations(features_list,2) ))    
        elif max_order == 3:
            features_list = [i for i in range(in_features)]
            features_list_copy = copy.deepcopy(features_list)
            features_list.extend(list( itertools.combinations(features_list_copy,2) ))
            features_list.extend(list( itertools.combinations(features_list_copy,3) )) 
        elif max_order == 4:
            features_list = [i for i in range(in_features)]
            features_list_copy = copy.deepcopy(features_list)
            features_list.extend(list( itertools.combinations(features_list_copy,2) ))
            features_list.extend(list( itertools.combinations(features_list_copy,3) )) 
            features_list.extend(list( itertools.combinations(features_list_copy,4) ))
        else:
            print("Order error")            
        
        
    tpnn_list = []
    for j in range(0,max_order):
        
        input_dim = j + 1
           
        bases_tpnn = TPNN(in_features=input_dim, 
                            num_tpnn=num_bases,
                            output_dim = num_multiclass, 
                            device= device, 
                            bin_function=bin_function,
                            monotone=monotone_list)
        tpnn_list.append(bases_tpnn) 
        
    return features_list, tpnn_list



## Evaluate stable Interpretability
def cal_UoC(all_data_loader, model_path, num_tpnn, max_order, in_features,device,regression,normalize = True,interaction_list = [],num_seed=10):

    device = device
    choice_function=nn_utils.entmax15
    bin_function=nn_utils.entmoid15


    num_Ks = num_tpnn    
    multiclass = 2
    
    
    if max_order == 1:
        num_component = in_features
    else:
        num_component = in_features + (in_features*(in_features-1))/2
    
    
    for data__ in all_data_loader:
        data__x = data__[:,:in_features].to(device) 
                        
    if regression:
        multiclass = 2

    if multiclass == 2:
        num_multiclass = 1
    else:
        num_multiclass = multiclass    
        
    var_features_list = []
    abs_features_list = []
    
    if len(interaction_list) == 0:
        component_list = [i for i in range(0,in_features)]
        
        if max_order == 2:
            component_list.extend(list( combinations(component_list,2)))
    else:
        component_list = interaction_list
        
    
    model_list = []
    
    for w in range(0,num_seed):
        
        if len(interaction_list) == 0:
            features_list_cs, tpnn_list = gen_tpnn(in_features,num_Ks,device,num_multiclass,max_order,choice_function, bin_function)
        else:
            features_list_cs, tpnn_list = gen_tpnn(in_features,num_Ks,device,num_multiclass,max_order,choice_function, bin_function,features_list=interaction_list)
        model = nn.Sequential(
            ANOVA_Ensemble(features_list_cs,tpnn_list,device),
            nn_utils.Lambda(lambda x:  x.mean(dim=1)),
        )
        model = model.to(device) 
        
        ##### Load model #####

        load_model_state =  torch.load(model_path + f"-{w}") 
                    
        model.load_state_dict(load_model_state)     
        model[0].training = False 
        
        model_list.append(model)
    

    for l in range(0,len(component_list)):
        feature_j = component_list[l]
        
        print(f"Componet : {feature_j} processing...")
        
        all_output_trial = torch.tensor([])
        for w in range(0,num_seed):           

            model = model_list[w]
            
            if len( data__x[:,feature_j].shape ) ==1 :
                output_comp = model[0][l](data__x[:,feature_j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu()/num_component
            else:
                
                if True:
                    input_new_data = torch.tensor([]).to(device)

                    for k in range(0,data__x.shape[0]):
                        
                        local_data = data__x[:,feature_j]
                        
                        local_data[:,0] = local_data[k,0]
                        
                        input_new_data = torch.concat([input_new_data,local_data])
                        
                    output_comp = model[0][l](input_new_data.float(),False).mean(dim=1).detach().cpu()/num_component
                else:
                    output_comp = model[0][l](data__x[:,feature_j].float(),False).mean(dim=1).detach().cpu()
                
            all_output_trial = torch.concat([all_output_trial,output_comp.reshape(1,-1)])
            
        if normalize == True:
                    
            for n in range(0,all_output_trial.shape[1]):
                if np.sqrt(np.sum(np.array(all_output_trial[:,n])**2))   != 0:
                    all_output_trial[:,n] = all_output_trial[:,n] / np.sum(np.array(all_output_trial[:,n])**2)

                        
        var_sum =0
        all_output_trial  = np.array( all_output_trial )
        for c in range(0,all_output_trial.shape[1]):
                
            var_sum += np.var(all_output_trial[:,c])
                
        var_sum /= all_output_trial.shape[1]
        
        abs_sum = 0
        for c in range(0,all_output_trial.shape[1]):
                
            abs_sum += np.mean( np.abs( all_output_trial[:,c] - np.mean(all_output_trial[:,c]) ) )
                
        abs_sum /= all_output_trial.shape[1]
                
        var_features_list.append(var_sum)
        abs_features_list.append(abs_sum)
        
    return var_features_list,abs_features_list



############ Figure Main shape function ############  

def make_fig(data_x,data_y, regression,max_order,num_tpnn,columns_list,model_path,device,cs=False,fig=True,init_test=False,init_random_seed=0,uniform_transform=False):

    if uniform_transform == True:
        data_x = 2*qt.fit_transform(data_x) -1
        
    train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=0)
    in_features = train_x.shape[1]
    
    if max_order == 1:
        num_component = in_features
    else:
        num_component = in_features + (in_features*(in_features-1))/2
        
    for w in range(0,10):
    
        # choice_function=nn_utils.entmax15
        # bin_function=nn_utils.entmoid15
        
        choice_function = nn.Softmax(dim=-1)
        bin_function = nn.Sigmoid() 
        

        num_Ks = num_tpnn    
        multiclass = 2
        
        
                            
        if regression:
            multiclass = 2

        if multiclass == 2:
            num_multiclass = 1
        else:
            num_multiclass = multiclass    
            
        features_list_cs, tpnn_list = gen_tpnn(in_features,num_Ks,device,num_multiclass,max_order,choice_function, bin_function)
        
        
        model = nn.Sequential(
                ANOVA_Ensemble(features_list_cs,tpnn_list,device),
                nn_utils.Lambda(lambda x:  x.mean(dim=1)))
        

        ##### Load model #####

        load_model_state =  torch.load(f"{model_path}-{w}")
        model.load_state_dict(load_model_state) 
        
        model = model.to(device)
        model.eval()
        
        if init_test == True:
            w = init_random_seed
            
        train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=w)
        val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_, test_size=0.66, random_state=0)
        
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        val_x = scaler.transform(val_x)
        
        if regression == True:
            test_y = (test_y - torch.mean(train__y))/torch.std(train__y)

        
        test_data = torch.cat([torch.tensor(test_x),test_y],dim=1)

        test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        
        
        if regression == True:
            
            test_loss = 0
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)   
                
                #print(test__x,test__x.shape)

                test_loss +=  torch.sum( (model(test__x.float()).flatten() - test__y.flatten())**2 )
                
            test_rmse = torch.sqrt( test_loss.cpu().detach()/len(test_x) )
            print(f"state {w} ||  test rmse : {test_rmse*torch.std(train__y)}")
            
        else:
            all_test__y = torch.tensor([])
            all_test__output = torch.tensor([])
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
                all_test__y = torch.concat([all_test__y,test__y.detach().cpu()])
                all_test__output = torch.concat([all_test__output,(model(test__x.float()).reshape(-1,1)).detach().cpu()]) 
                
            test_measure = roc_auc_score(all_test__y,all_test__output)  
            print(f"state {w} || test auc : {test_measure}")
            
        
        if fig == True:
            main_list = []
            
            for f in range(0,len(features_list_cs)):
                
                if type( features_list_cs[f] ) == int:
                    
                    main_list.append(features_list_cs[f])
            
            
            if cs:
                max_feature = np.min([len(main_list),11])
            else:
                max_feature = np.min([in_features,11])
            
            
            f, axes = plt.subplots(1, max_feature, sharex=False, sharey=False)
            f.set_size_inches((25, 2))  
            
            f.text(0.09, 0.5, "Output Contribution", va='center', rotation='vertical') 
            
            y_max_list = []
            y_min_list = []
            for j in range(0,max_feature):
                y_max_list.append( np.max(np.array(  model[0][j](test__x[:,j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu() )/num_component ) )
                y_min_list.append( np.min(np.array(  model[0][j](test__x[:,j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu() )/num_component ) )
            
            
            y_max = np.max(y_max_list) + np.max(y_max_list)/10
            y_min = np.min(y_min_list) + np.min(y_min_list)/10
            
            sns.set(font_scale=1.0)
            for i in range(0,max_feature):
                
                axes[i].set(xlabel = columns_list[i])
            
                #axes[i].set(ylabel = "Output contribution")
                
                axes[i].set_ylim([y_min,y_max])
            
                scatter_x = np.array( test__x[:,i].detach().cpu() )

                scatter_y = np.array(  model[0][i](test__x[:,i].reshape(-1,1).float(),False).mean(dim=1).detach().cpu()/num_component  ) 
                
                #scatter = pd.DataFrame([scatter_x,scatter_y],index = [f"variable {i}",f"f {i}"]).T
                sns.lineplot(  x=scatter_x.flatten(),y=scatter_y.flatten(),ax=axes[i],color = "blue")
                sns.set(font_scale=1.0)
            f.show()
