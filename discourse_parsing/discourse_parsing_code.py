# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:46:12 2022

@author: artst
"""

import os, glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import confusion_matrix, average_precision_score
import statsmodels.api as sm


#Discourse Parsing Experiment

def process_path(path,t='ph',id__='_gen',generated=True):

    allfiles=glob.glob(os.path.join(path,'*.brackets'))
    
    dfs_=[]
    
    for file in allfiles:
    
        c=file.split('\\')[-1].split('.')[0]
        id_=c+id__
        try:
            df=pd.read_csv(file,header=None)
            df['Id']=id_
            df['type']='ph'   #ML or physics abstract
            df[0]=df[0].apply(lambda x: x[2:])
            df[1]=df[1].apply(lambda x: x[:-1])
            df[2]=df[2].apply(lambda x: x[1:].replace('\'',''))
            df[3]=df[3].apply(lambda x: x[1:-1].replace('\'',''))
        except:  #tree is empty
            df=pd.DataFrame()
            df=df.append({0: np.nan,1: np.nan,2: np.nan,3: np.nan, 'Id': id_, 'type': 'ph'}, ignore_index=True)
    
        dfs_.append(df)
    
    
    dfs_=pd.concat(dfs_)
    dfs_['gen']=1*generated
    

    return dfs_


#read the parsed tree data
dfs_generated=process_path(r'F:\arxiv code\generated\generated',t='ph',id__='_gen',generated=True)
dfs_true=process_path(r'F:\arxiv code\true\true',t='ph',id__='_true',generated=False)
dfs_true7500=process_path(r'F:\arxiv data\rsts\true7500\true7500',t='ML',id__='_true7500',generated=False)
dfs_gen7500_09=process_path(r'F:\arxiv data\rsts\generated7500_0.9\generated7500_0.9',t='ML',id__='_gen7500_09',generated=True)

dfs=pd.concat((dfs_true,dfs_generated,dfs_gen7500_09,dfs_true7500))


#Extract features from trees
# =============================================================================
# Features:
# size (number of nodes), number of unique nucleuses, satellites (2), number of unique relations (3),
# number of edus = max (1)
# number of same in (0)
# =============================================================================


dfs[0]=pd.to_numeric(dfs[0],errors='coerce')
dfs[1]=pd.to_numeric(dfs[1],errors='coerce')

#extract unique relations
satnuc=dfs[2].dropna().unique().tolist()
rels=dfs[3].dropna().unique().tolist()


groups=[]
for i,group in dfs.groupby('Id'):

    feat={}
    if group.shape[0]==1:  #empty tree
        if np.isnan(group[0].item()):

            feat['ML']=1
            if group['type'].item()=='ph':
                feat['ML']=0
            feat['span_1']=0
            feat['span_2']=0
            feat['span_3']=0
            feat['span_4']=0
            
            feat['n_edu']=0

            feat['size']=0
            for col in satnuc+rels:
                feat[col]=0
    else:
            #populated tree
            span=group[1]-group[0]        
            span=span.loc[span>0]
           
            feat['ML']=1
            if group['type'].iloc[0]=='ph':
                feat['ML']=0
            feat['span_1']=(span==1).sum()
            feat['span_2']=(span==2).sum()
            feat['span_3']=(span==3).sum()
            feat['span_4']=(span==4).sum()

            feat['n_edu']=group[1].max()

            feat['size']=group.shape[0]
            for col in satnuc:
                feat[col]=(group[2]==col).sum()
            for col in rels:
                feat[col]=(group[3]==col).sum()            
    
    feat['gen']=group['gen'].iloc[0]

    groups.append(feat)


A=pd.DataFrame(groups)

A=A.sample(frac=1,random_state=2022)

AML=A.loc[A['ML']==1]  #ML abstracts
APH=A.loc[A['ML']!=1]  #Physics abstracts


#In-sample L1 regulatized Logistic Regression
 
# defining the dependent and independent variables
Xtrain = APH.iloc[:,1:-1]
ytrain = APH.iloc[:,-1]

logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit_regularized(alpha=0.03)
print(logit_model.summary())

Xtrain = AML.iloc[:,1:-1]
ytrain = AML.iloc[:,-1]
logit_model = sm.Logit(ytrain, sm.add_constant(Xtrain)).fit_regularized(alpha=0.2)
print(logit_model.summary())


#logit_model.summary2().tables[0].to_csv(r"F:\arxiv code\logit_summary_stat.csv")
#logit_model.summary2().tables[1].to_csv(r"F:\arxiv code\logit_summary_stat1.csv")




#Out-of-sample analysis
n_trials=1000

label='gen'
X=list(A)[:-1]
N=A.shape[0]

#randomly split into train-val-test, A was shuffled earlier
traindf=A.iloc[:int(0.7*N),:]
validdf=A.iloc[int(0.7*N):int(0.85*N),:]
testdf=A.iloc[int(0.85*N):,:]


Xtrain = traindf[X].values
Xval = validdf[X].values
Xtest = testdf[X].values


Ytrain = traindf[label].values
Yval = validdf[label].values
Ytest = testdf[label].values


#consider RF vs Logistic, hyperparams selected via optuna, score to maximize is f1

def objective(trial):
   
   classifier_name = trial.suggest_categorical("regressor", ["LogisticRegression", "RandomForest"])


   if classifier_name == 'LogisticRegression':
       ridge_alpha = trial.suggest_float("ridge_alpha", 0.1, 10000, log=False)
       classifier_obj = LogisticRegression(C=ridge_alpha, class_weight='balanced')
   elif classifier_name == 'RandomForest':
       rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
       rf_max_depth = trial.suggest_int("rf_max_depth", 2, 150, log=False)
       rf_max_feat = trial.suggest_int("rf_max_feat", 2, len(X), log=False)

       classifier_obj=RandomForestClassifier(
       max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_feat, random_state=2022, class_weight='balanced')
  

   classifier_obj.fit(Xtrain,Ytrain)
   yhat=classifier_obj.predict(Xval)

   tn, fp, fn, tp = confusion_matrix(Yval,yhat).reshape(-1,1)
   p=tp/(tp+fp)
   r=tp/(tp+fn)
   score=2*p*r/(r+p)
   if score!=score:
       score=0
   return score
   

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, n_jobs=8)


best_params=study.best_params
   
if best_params['regressor'] == 'LogisticRegression':
   ridge_alpha = best_params["ridge_alpha"]
   model = LogisticRegression(C=ridge_alpha, class_weight='balanced')
elif best_params['regressor'] == 'RandomForest':
    rf_n_estimators = best_params["rf_n_estimators"]
    rf_max_depth = best_params["rf_max_depth"]
    rf_max_feat = best_params["rf_max_feat"]

    model=RandomForestClassifier(
    max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_feat, random_state=2022, class_weight='balanced')

#train best model
model.fit(Xtrain, Ytrain)

#evaluate best model on training set
pred=model.predict(Xtrain)
tn, fp, fn, tp = confusion_matrix(Ytrain,pred).reshape(-1,1)
trainprecision=(tp/(tp+fp)).item()
trainrecall=(tp/(tp+fn)).item()
trainavprec=average_precision_score(Ytrain,model.predict_proba(Xtrain)[:,1])
trainf1=2*trainprecision*trainrecall/(trainrecall+trainprecision)
#evaluate on validation set
pred=model.predict(Xval)
tn, fp, fn, tp = confusion_matrix(Yval,pred).reshape(-1,1)
valprecision=(tp/(tp+fp)).item()
valrecall=(tp/(tp+fn)).item()
valavprec=average_precision_score(Yval,model.predict_proba(Xval)[:,1])
valf1=2*valprecision*valrecall/(valrecall+valprecision)
valacc=np.mean(pred==Yval)

#evaluate on test set
pred=model.predict(Xtest)
tn, fp, fn, tp = confusion_matrix(Ytest,pred).reshape(-1,1)
testprecision=(tp/(tp+fp)).item()
testrecall=(tp/(tp+fn)).item()
testavprec=average_precision_score(Ytest,model.predict_proba(Xtest)[:,1])
testf1=2*testprecision*testrecall/(testrecall+testprecision)
testacc=np.mean(pred==Ytest)

#report results
res=pd.DataFrame([(label,trainprecision,trainrecall, trainavprec, trainf1,
valprecision,valrecall, valavprec, valf1,
testprecision,testrecall, testavprec, testf1,
                best_params,valacc,testacc)])
res.columns=['label','trainprecision','trainrecall',' trainavprec',' trainf1',
    'valprecision','valrecall',' valavprec',' valf1',
    'testprecision','testrecall',' testavprec',' testf1',
                    'best_params','valacc','testacc']

#res.to_csv(r"F:\arxiv code\logit_sklearn.csv", index=False)

