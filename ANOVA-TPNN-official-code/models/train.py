import numpy as np 
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from torch import optim
import pickle

from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import roc_auc_score

from .tpnn import TPNN
from models import ensemble
from models import nn_utils
from models import utils

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import QuantileTransformer


def Trainer(data_x,
            data_y, 
            num_Ks , 
            max_order , 
            device, 
            model_path , 
            measure ,
            regression,
            random_state,
            bin_function = nn.Sigmoid(),
            multiclass = 2 ,
            lr_rate =0.001 , 
            epoch_num=1000 , 
            num_train_batch = 2048,
            num_test_batch = 1024 ,
            init_train=True,
            reg_lambda=0.0,
            features_list="all",
            monotone_list =[],
            uniform_transform=True):
    
    in_features = data_x.shape[1]
    qt = QuantileTransformer(n_quantiles=data_x.shape[0],
                             random_state=0,
                             output_distribution='uniform')
    
    if regression:
        multiclass = 2

    
    if multiclass == 2:
        class_num = 1
        
        
    if regression == True:
        global_opt_loss = np.inf 
    else:
        global_opt_loss = -np.inf
    

    
    ### generating model
    
    features_list, tpnn_list = ensemble.gen_tpnn(in_features=in_features,
                                            num_Ks=num_Ks,
                                            device=device,
                                            num_multiclass=class_num,
                                            max_order=max_order,
                                            bin_function=bin_function,
                                            features_list=features_list,
                                            monotone_list=monotone_list)
    
    model = nn.Sequential(
        ensemble.ANOVA_Ensemble(features_list,tpnn_list,device),
        nn_utils.Lambda(lambda x:  x.sum(dim=1)),
    )
    
    model = model.to(device)    
    
    train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=random_state)
    val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_, test_size=0.66, random_state=0)
    
    if uniform_transform == True:
        train_x = 2*qt.fit_transform(train_x) -1
        test_x = 2*qt.fit_transform(test_x) -1
        val_x = 2*qt.fit_transform(val_x) -1
    else:
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        val_x = scaler.transform(val_x)

    if regression == True:
        train_y = (train__y - torch.mean(train__y))/torch.std(train__y)
        test_y = (test_y - torch.mean(train__y))/torch.std(train__y)
        val_y = (val_y - torch.mean(train__y))/torch.std(train__y)

    train_data = torch.cat([torch.tensor(train_x),train_y],dim=1)
    test_data = torch.cat([torch.tensor(test_x),test_y],dim=1)
    val_data = torch.cat([torch.tensor(val_x),val_y],dim=1)

    train_dataloader = DataLoader(train_data, batch_size=num_train_batch, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=num_test_batch, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=num_test_batch, shuffle=True)  


    ############## Initalize ###############
    for train__ in train_dataloader:
        init_x,init_y = train__[:,:in_features].float() , train__[:,in_features].float()
        
        init_x,init_y = init_x.to(device),init_y.to(device)
        break
        
    if init_train == True:    
        with torch.no_grad():
            model[0].training = True
            model[0](init_x,inital=True)
        
        
    optimizer = optim.Adam(list(model.parameters()), lr=lr_rate,betas=(0.95, 0.998))
    
    if regression:
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    
       
    for epoch in range(epoch_num):
        
        model.train()
        loss_sum = 0
        model[0].training = True
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_ = (batch[:,0:in_features], batch[:,in_features:])
            batch_x , batch_y = batch_
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            
            if multiclass == 2:
                loss = criterion(output.flatten(),batch_y.flatten()).sum()
                
            else:
                print("multiclass > 3 error")
            
            if reg_lambda > 0.0 :
                loss += reg_lambda*torch.mean( (model[0](batch_x))**2)
            
            loss.backward() 
            optimizer.step() 
            loss_sum += loss.item()
        
        if regression:
            epoch_RMSE = np.sqrt(loss_sum/len(train_data))*torch.std(train__y)
        else:
            epoch_RMSE = loss_sum/len(train_data)


        model.eval()        
        model[0].training = False
        model[0].model_save_id_constants()
               
                
        if regression == True:  
            val_loss = 0
            for val__ in val_dataloader:
                val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
            
                val_loss += torch.sum( (model(val__x.float()).flatten() - val__y.flatten())**2 ).cpu().detach()
            val_rmse = np.sqrt(  (val_loss.cpu().detach() )/len(val_data)  )*torch.std(train__y)
            

            if val_rmse < global_opt_loss:
                
                test_loss =0
                all_test__output = torch.tensor([])
                
                for test__ in test_dataloader:
                    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)   
                
                    all_test__output = torch.concat([all_test__output, model(test__x.float()).cpu().detach()  ])
                    
                    test_loss += torch.sum( (model(test__x.float()).flatten() - test__y.flatten())**2 ).cpu().detach()
                test_local_measure = np.sqrt(  (test_loss.cpu().detach() )/len(test_data)  )*torch.std(train__y)
                    
                opt_epoch = epoch
                global_opt_loss = val_rmse
                
                test_measure = test_local_measure
                
                print(f"Epoch : {epoch} || train rmse : {round(epoch_RMSE.item(),4) } , val rmse : {round(val_rmse.item(),4)}, test rmse : {round(test_measure.item(),4)}")
    
                if model_path != None:
                    torch.save(model.state_dict(),model_path)
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            # early stopping.
            if epoch - opt_epoch == 1000:
                return model,test_measure ,all_test__output
         
            
        else:
               
            if measure == "acc":
                val_hit =0
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    
                    val_logit = model(val__x.float()).flatten()  >= 0 
                    true_val_logit = val__y.flatten() >=0
                        
                    val_hit += torch.sum( val_logit== true_val_logit )                       
                
                val_measure = val_hit/ len( val_data )
                
            elif measure == "auc":
                    
                all_val__y = torch.tensor([])
                all_val__output = torch.tensor([])
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    all_val__y = torch.concat([all_val__y,val__y.detach().cpu()])
                    all_val__output = torch.concat([all_val__output,(model(val__x.float()).reshape(-1,1)).detach().cpu()])
                      
                val_measure = roc_auc_score(all_val__y,all_val__output )
                                    
            
            if val_measure >= global_opt_loss :
 
                if measure == "acc":
                    test_hit = 0
                    for test__ in test_dataloader:
                        test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)  
                        
   
                        test_logit = model(test__x.float()).flatten()  >= 0 
                        true_test_logit = test__y.flatten() >=0
                            
                        test_hit += torch.sum( test_logit== true_test_logit )                                       

                    test_candi_measure = test_hit/ len(test_data)
                    
                elif measure == "auc":
                    
                    all_test__y = torch.tensor([])
                    all_test__output = torch.tensor([])
                    for test__ in test_dataloader:
                        test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
                        all_test__y = torch.concat([all_test__y,test__y.detach().cpu()])
                        all_test__output = torch.concat([all_test__output,(model(test__x.float()).reshape(-1,1)).detach().cpu()]) 
                
  
                    test_candi_measure = roc_auc_score(all_test__y,all_test__output )            
                                

                opt_epoch = epoch 
                global_opt_loss = val_measure
                test_measure = test_candi_measure
                
                print(f"Epoch : {epoch}  || train rmse : {round(epoch_RMSE,4) } , val {measure} : {round(val_measure,4)}, test {measure} : {round(test_measure,4)}") 
                if model_path != None:
                    torch.save(model.state_dict(),model_path) 
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            print(f"Epoch : {epoch} || train rmse : {epoch_RMSE }")
            if epoch - opt_epoch == 1000:
                return model,test_measure,all_test__output
                    
    return model,test_measure, all_test__output
    