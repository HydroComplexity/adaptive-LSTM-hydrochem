
########################################Multiprocessing#########################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
import torch
from torch import nn
from sklearn.metrics import mean_squared_error as MSE
from torch.multiprocessing import Process
import os
from lstm_classes import Sequence, RealNVP
import sys
import utils
from tqdm import tqdm
import random

# tensor.to("cuda") device = torch.device("cuda:0") --> tensor.to(device)
device = sys.argv[1]
if device=='cpu':
    device=device
elif 'cuda' in device:
    device = torch.device(device)
else:
    raise NotImplementedError
    device=device

# device  = 'cpu'
# print('running on CPU')

seed=42
torch.manual_seed(seed)


def allvar(training_set,var_list,tau,H,num_epochs,fr,inputvar,invar_list,learning_rate,header,seq_length,num_classes,device,mgate,fluxgate,combgate,fold,static,resample,gi,g,eps,epsx):

    process_id=os.getpid()
    print('######################################')
    print('Process ID'+str(process_id))
    print('######################################')
    print(mgate)

    mse_mlstmts = np.empty (len (H))    #np.empty ((len (tau), len (H)))
    mse_mlstmtr = np.empty (len (H))
    hk = 0              #initialization of hidden size vector

    training_seti=training_set[1:,:]  #tar, Q, tar, dependent variablei

    slp=utils.getgrad(training_set[:,1])
    slp1=utils.getgrad(training_set[:,2])
    training_seti=np.hstack((training_seti,slp.reshape((len(slp),1)),slp1.reshape((len(slp1),1))))  #tar, Q, tar, dependent variablei, gradient of Q, gradient of tar
    # if fluxgate=='yes':
    flux=training_set[:,1]*training_set[:,2]
    slp2 = utils.getgrad (flux)
    training_seti = np.hstack ((training_seti, slp2.reshape ((len (slp2), 1))))  # tar, Q, tar, dependent variablei, gradient of Q, grad tar, gradient of tar flux

    training_set=training_seti

    slpid=slp
    slp2id=slp2
    for i in range(len(slp)):
        if slp[i]<=0+eps and slp[i]>=0-eps:
            slpid[i]=0
        if slp[i]>0+eps:
            slpid[i]=1
        if slp[i]<0-eps:
            slpid[i]=-1
    for i in range(len(slp2)):
        if slp2[i]<=0+epsx and slp2[i]>=0-epsx:
            slp2id[i]=0
        if slp2[i]>0+epsx:
            slp2id[i]=1
        if slp2[i]<0-epsx:
            slp2id[i]=-1


    for hidden_size in tqdm(H):

        nk=tau
        input_size1 = len(inputvar)*(nk)+1        #number of inputs+1
        diffvar1=len(inputvar)*(nk)+1


        input_size = len(inputvar)*(nk)+1        #number of inputs+1


        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)  #normalisation
        if combgate=='yes':
            training_data = np.hstack ((training_data, slpid.reshape (len (slpid), 1),slp2id.reshape (len (slp2id), 1)))  # added flowid and flux id


        x, y = utils.sliding_windows(training_data, seq_length)

        train_size = int(len(y) * fr)
        test_size = len(y) - train_size

        trainX = torch.Tensor(np.array(x[:train_size,:,:])).to(device)
        trainY = torch.Tensor(np.array(y[:train_size,:])).to(device)

        testX = torch.Tensor(np.array(x[train_size:,:,:])).to(device)
        testY = torch.Tensor(np.array(y[train_size:,:])).to(device)

        if __name__ == '__main__' :
            input = trainX[:,:,input_size1+num_classes-diffvar1:]                 #Q, tar, dependent variablei, gradient of Q, gradient of tar  #trainX row from 3 and column till last -1
            if combgate == 'no' :
                if mgate == 'regLSTM' :
                    print ('Not using flow gate regLSTM')
                else :
                    print ('Using flow gate mLSTM')
                if fluxgate == 'no' :
                    print('Not using flux gate')
                else :
                    print('Using flux gate')
            else :
                print ('combnining the effects of the flow, flux gates')

            target = trainY[:,0:num_classes]                                       #trainY row from 3rd and column from 1 to end
            test_input = testX[:,:,input_size1+num_classes-diffvar1:]                                    #till row 3 and column till last -1
            test_target = testY[:,0:num_classes]                                  # till row 3 and column from 1 to last


            stshy=0
            seq = Sequence(input_size,hidden_size,seq_length,num_classes,mgate,fluxgate,combgate,inputvar,nk,stshy,LSTM=True,custom=True,device=device,).to(device)

            seq.float ()
            criterion = nn.MSELoss().to(device)
            optimizer = optim.Adam(seq.parameters (), lr=learning_rate)

            # begin training
            lss = np.ones ((num_epochs, 1))

            input,target=input.cpu(),target.cpu()
            shflinput,shfltarget,permutation = utils.shuffle_in_unison(input.numpy(),target.numpy())  # random shuffling along the 1st axis

            input = torch.tensor(shflinput).to(device)
            target = torch.tensor (shfltarget).to (device)

            losshk = np.zeros (num_epochs)
            for i in range (num_epochs):
                seq.train()
                optimizer.zero_grad ()

                out = seq(input)
                loss = criterion (out, target)
                losshk[i] = loss.item ()
                if i % 2 == True :
                    print ('epoch:', i, 'loss:', loss.item ())
                if np.isnan(loss.item()).any():
                    break
                if loss.item () <= .0009 :
                    break
                lss[i] = loss.item ()
                if i >= 2 :
                    lss1 = lss[i] - lss[i - 2]
                    if abs (lss1) <= 0.0005 :
                        break
                loss.backward ()
                optimizer.step ()


            #testing the model
            with torch.no_grad () :  # no gradient calculations
                seq.eval ()
                future = 0
                torch.tensor(future).to(device)
                pred = seq (test_input, future=future)
                pred_train = seq (input, future=future)
                loss1 = criterion (pred_train, target)
                loss = criterion (pred, test_target)
                print ('train loss:', loss1.item ())
                print ('test loss:', loss.item ())

            pred_train,target=pred_train.cpu(),target.cpu()
            inversepred_train, inversetarget=utils.inverse_shuffle_in_unison(pred_train.numpy(),target.numpy(),permutation)
            pred_train = torch.tensor(inversepred_train).to(device)
            target = torch.tensor (inversetarget).to (device)

            yy = pred.detach().cpu().numpy()
            tr = pred_train.detach().cpu().numpy ()
            target=target.cpu()
            test_target=test_target.cpu()
            #result
            #lencorr1 is for the inverse transformation
            train_trfmp=utils.lencorr1(tr,len(y))
            train_trfmo=utils.lencorr1(target,len(y))
            test_trfmp=utils.lencorr1(yy,len(y))
            test_trfmo=utils.lencorr1(test_target,len(y))


            ntrain_predict=train_trfmp[:len(tr)]
            ntrainY_plot=train_trfmo[:len(target)]
            ntest_predict=test_trfmp[:len(yy)]
            ntestY_plot=test_trfmo[:len(test_target)]

            if np.isnan(ntrain_predict).any():
                mse_mlstmtr[hk]=11111
                mse_mlstmts[hk]=11111
                print ('nan in prediction')
            else:
                mse_mlstmtr[hk] = MSE (ntrainY_plot, ntrain_predict, multioutput='raw_values')
                mse_mlstmts[hk]=MSE(ntestY_plot,ntest_predict,multioutput='raw_values')


            #prediction is happening at different time
            #lencorr2 is correcting the time step displacement because of the tau
            ntrain_ip=utils.lencorr2(ntrain_predict,train_size,seq_length)
            ntrain_io=utils.lencorr2(ntrainY_plot,train_size,seq_length)
            ntest_ip=utils.lencorr2(ntest_predict,test_size,0)
            ntest_io=utils.lencorr2(ntestY_plot,test_size,0)


        if hk==0:
            nprediction1 = np.concatenate ((ntrain_ip, ntest_ip))
            nobserved1 = np.concatenate ((ntrain_io, ntest_io))
            df2 = pd.DataFrame (nprediction1)
            df3 = pd.DataFrame (nobserved1)
            df8= pd.DataFrame (losshk)
            if combgate == 'no' :
                df2.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +'_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nprediction.csv')
                df3.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +'_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nobserved.csv')
                df8.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +'_fluxgate_' + fluxgate + resample + '_' + str (g) + '_gloss.csv')
            if combgate == 'yes' :
                df2.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) +'_epsfx_'+str(epsx)+ '_' + resample + '_' + str (g) + '_nprediction.csv')
                df3.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_nobserved.csv')
                df8.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_gloss.csv')
        else:
            nprediction1 = np.concatenate ((ntrain_ip, ntest_ip))
            nobserved1 = np.concatenate ((ntrain_io, ntest_io))
            if combgate == 'no' :
                predfile1 = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate + '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nprediction.csv')
                obsfile1 = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +'_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nobserved.csv')
                losk = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate + '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_gloss.csv')
            if combgate == 'yes' :
                predfile1 = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_nprediction.csv')
                obsfile1 = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_nobserved.csv')
                losk = pd.read_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_gloss.csv')
            predfile = predfile1.iloc[:, 1 :].values
            obsfile = obsfile1.iloc[:, 1 :].values
            predfile = np.hstack ((predfile, nprediction1))
            obsfile = np.hstack ((obsfile, nobserved1))

            losk = losk.iloc[:, 1 :].values
            losk = np.hstack ((losk, losshk.reshape (num_epochs, 1)))
            df2 = pd.DataFrame (predfile)
            df3 = pd.DataFrame (obsfile)
            df8 = pd.DataFrame (losk)

            if combgate == 'no' :
                df2.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +  '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nprediction.csv')
                df3.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +  '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_nobserved.csv')
                df8.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate + '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_gloss.csv')
            if combgate == 'yes' :
                df2.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_nprediction.csv')
                df3.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) +'_epsfx_'+str(epsx)+ '_' + resample + '_' + str (g) + '_nobserved.csv')
                df8.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_gloss.csv')

        if hk+1<len(H):
                print('######################################################################################')
                print("####################################### Hidden Size ",str(H[hk]) , "#####################################")
                print("####################", "  ", "Now starting Hidden Size =",str(H[hk+1]), "#######################")
                print("######################################################################################")
        hk=hk+1



    df6=pd.DataFrame(np.transpose([mse_mlstmtr],(0,1)))
    df7=pd.DataFrame(np.transpose([mse_mlstmts],(0,1)))

    import time
    timestr = time.strftime ("%Y%m%d-%H%M%S")
    print(timestr)

    if combgate == 'no' :
        df6.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate + '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_mse_traindata.csv', header=header)
        df7.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_' + mgate +  '_fluxgate_' + fluxgate + resample + '_' + str (g) + '_mse_testdata.csv', header=header)
    if combgate == 'yes' :
        df6.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_mse_traindata.csv', header=header)
        df7.to_csv (fold + var_list + '_' + str (invar_list) + '_fr_' + str (fr) + '_' + str (gi) + '_' + static + '_seq_' + str (seq_length) + '_combgate_' + combgate + '_eps_' + str (eps) + '_epsfx_'+str(epsx)+'_' + resample + '_' + str (g) + '_mse_testdata.csv', header=header)


#####################################################################################################################

#########################################
############ data input #################
#########################################

seq_length = [5]
num_classes = 1  # number of outputs

fr=[0.7]
num_epochs =40


fluxgate=['yes','no']
fluxgateid=0


method=['regLSTM','mLSTM(tanh)']
ind=1
combgateid=0


epsfl=[0,0.01,0.001,0.0005,0.0001,0.00001]
epsfx=[0,0.01,0.001,0.0005,0.0001,0.00001]


H = [25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,98,100,120,130]
header = [f'h={hval}' for hval in H]


lr =.01
print('lr='+str(lr))


################ Orgeval data ###########
var=[1,2,3,4,5,6,7]
var_list=['Mg','K','Ca','Na','SO4','NO3','Cl']

# 2var
inputvar12=[[8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7]]
invar_list12=[['Q','Mg'],['Q','K'],['Q','Ca'],['Q','Na'],['Q','SO4'],['Q','NO3'],['Q','Cl']] #test

inputvar0=[inputvar12]
invar_list0=[invar_list12]


fol=['2var/']
folder0='/home/.../output/'

dropoutp=[0.1,0.2]

tau_max = 2
nk=1

tau_s=[tau_max]
print(static)
print('Output='+str(var_list))
print('learning rate='+str(lr))
print('seq length='+str(seq_length))
print('lenth of fr '+str(len(fr)))
print(str(fr))
print('Tau '+str(tau_max))
print('method='+method[ind])
print(folder0)

resfilename=['0.5hr']
res=0
print('res='+resfilename[res])

address='//home/.../orgeval_'+resfilename[res]+'_processed'
sheetname='orgeval'
print(address)

resample='_'+resfilename[res]

rkey=3
stpt,gi1,gf1=1,1,18700

Gi=[1]

gi,gf=gi1,gf1
print('gi is                                                    '+str(gi))
print('gf is                                                    '+str(gf))
ep,epx=0,0
for ep in range(len(epsfl)):
    print('epsfl=',str(epsfl[ep]))
    for epx in range(len(epsfx)):
        print ('epsfx=', str (epsfx[epx]))
        for k1 in range(len(fol)):
            for i in range (len (var)) :
                print('**************************')
                print(var_list[i])
                print('**************************')
                g=0
                print (invar_list0[k1])
                inputvar1=inputvar0[k1]
                invar_list1=invar_list0[k1]
                folder=folder0+fol[k1]
                if len (inputvar1[i]) == 2 :
                    l, m, n= utils.dataretrive (var[i], inputvar1[i], tau_max, address+'.csv', sheetname,stpt,gi,gf)
                    u, v = np.hstack ((l, m[:,0].reshape((len(m),1)), n[:,0].reshape((len(m),1)))), l



                for k in range(len(seq_length)):
                    process1 = Process (target=allvar, args=(u,inn[innid],ffn[ffnid],ct[ctid],var_list[i], nk, H, num_epochs, fr[0], inputvar1[i],
                                                             invar_list1[i], lr, header, seq_length[k], num_classes, device,method[ind],
                                                             fluxgate[fluxgateid],fluxgate[combgateid],folder,static,resample,gi,g,epsfl[ep],epsfx[epx]))
                    process1.start ()

            print ("multiprocessing completed only hidden state")