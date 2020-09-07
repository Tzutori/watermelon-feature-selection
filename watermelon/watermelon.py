'''
watermelon feature selection
Copyright (C) 2020 Xiang Xie xiang.xie.china@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import math
from scipy.stats import iqr
import time
from multiprocessing import Pool
from itertools import repeat
from scipy.special import comb
import logging

class watermelon():
    
    def __init__(self):
        pass
    
    
    def getErrorRate(self,data,other_data):
        max_value=np.max([np.max(data),np.max(other_data)])
        min_value=np.min([np.min(data),np.min(other_data)])
        
        data=data.reshape(-1,1)
        other_data=other_data.reshape(-1,1)
        
        len_data=len(data)
        len_other_data=len(other_data)
        #Silverman's rule of thumb to determine the bandwidth
        bw=1.06*np.min([np.std(data),iqr(data)/1.34])*len_data**(-0.2)
        bw_other=1.06*np.min([np.std(other_data),iqr(other_data)/1.34])*len_other_data**(-0.2)

        bw=max(self.min_kde_bw,bw)
        bw_other=max(self.min_kde_bw,bw_other)
        
        kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(data)
        dens=np.exp(kde.score_samples(np.linspace(min_value, max_value,self.kde_bins)[:, np.newaxis]))
        dens=dens/np.sum(dens)
        kde=KernelDensity(kernel='gaussian',bandwidth=bw_other).fit(other_data)
        dens_other=np.exp(kde.score_samples(np.linspace(min_value, max_value,self.kde_bins)[:, np.newaxis]))
        dens_other=dens_other/np.sum(dens_other)
        
        p_data=len_data/(len_data+len_other_data)
        p_other_data=1-p_data
        
        ber=0

        for i in np.arange(self.kde_bins):
            if dens[i]>dens_other[i]:
                ber+=p_other_data*dens_other[i]
            else:
                ber+=p_data*dens[i]
       
        return ber

    def chunks(self,l, n):
        #Yield successive n-sized chunks from l
        for i in range(0, len(l), n):
            yield l[i:i + n]

  
    def getErrorRateParallel(self,data,labels,classes):
        n_class=len(classes)
        n_subfeature=data.shape[1]
        if not self.ovo:
            result=np.zeros((n_class,n_subfeature),dtype='float')
            for index_feature in np.arange(n_subfeature):
                #calculate the error rate for each class
                for class_index in np.arange(n_class):
                    clazz=classes[class_index]
                    data_this_class=data[labels==clazz][:,index_feature]
                    data_other_class=data[labels!=clazz][:,index_feature]
                    result[class_index,index_feature]=self.getErrorRate(data_this_class,data_other_class)
            return result
        else:
            combis=comb(n_class,2,True)
            result=np.zeros((combis,n_subfeature),dtype='float')
            for index_feature in np.arange(n_subfeature):
                #calculate the error rate for each class
                cur_index=0
                for class_index in np.arange(n_class):
                    for other_class_index in np.arange(class_index+1,n_class):
                        clazz=classes[class_index]
                        other_clazz=classes[other_class_index]
                        data_this_class=data[labels==clazz][:,index_feature]
                        data_other_class=data[labels==other_clazz][:,index_feature]
                        result[cur_index,index_feature]=self.getErrorRate(data_this_class,data_other_class)
                        cur_index+=1
            return result
    
    
    def getNmiParallel(self,this_data,other_data):
        result=np.zeros((this_data.shape[1],other_data.shape[1]))
        for this_index in np.arange(this_data.shape[1]):
            current_data=this_data[:,this_index]
            for other_index in np.arange(other_data.shape[1]):
                current_other_data=other_data[:,other_index]
                result[this_index,other_index]=normalized_mutual_info_score(current_data,current_other_data,average_method='arithmetic')
        return result
    
    def calculateFeatureScore(self,score_list,score_of_select_feature,spearman_coe,nmi_coe,result):
        #shape of (n_class,n_to_evaluate_feature,n_selected_feature+1)
        result_new_score=np.zeros((score_list.shape[0],score_list.shape[1],len(result)+1))
        result_new_feature_index=-1
        #index of the features of current to-evaluate feature subset
        new_feature_index_map=score_list.columns.astype('int').values
        #scores of to-evaluate features, will be updated iteratively
        new_feature_score=score_list.to_numpy(copy=True)
        #iter over selected feature and update scores
        for selected_feature_index in range(score_of_select_feature.shape[1]):
            #make matrix to speed up score calculation, shape is (n_class,n_to_evaluate_feature)
            tiled_selected_score=np.tile(score_of_select_feature[:,selected_feature_index].reshape(-1,1),(1,score_list.shape[1]))
            #update according to nmi and cor
            #use largest cor between selected features and each to-evaluate feature, because it is dominant. Shape is (1, n_to_evaluate_feature)
            cor_max=np.max(spearman_coe[np.ix_(result,new_feature_index_map)],axis=0).reshape(1,-1)
            #Shape is (1, n_to_evaluate_feature)
            cur_nmi=nmi_coe[result[selected_feature_index],new_feature_index_map].reshape(1,-1)
            #positive values: selected feature should be updated; negative values: new feature should be updated
            score_diff=tiled_selected_score-new_feature_score
            update_selected_feature_score=score_diff*(score_diff>0)
            update_new_feature_score=score_diff*(score_diff<=0)*(-1)
            #update selected features
            temp_selected_feature_score=tiled_selected_score-update_selected_feature_score*self.activate_function(cur_nmi,self.th_nmi)
            temp_selected_feature_score+=(tiled_selected_score-temp_selected_feature_score)*self.activate_function(cor_max,self.th_cor)
            temp_selected_feature_score=np.clip(temp_selected_feature_score,a_min=None,a_max=tiled_selected_score)
            #update new features
            new_feature_max_score=np.ones(new_feature_score.shape)
            new_feature_score-=update_new_feature_score*self.activate_function(cur_nmi,self.th_nmi)
            new_feature_score+=(new_feature_max_score-new_feature_score)*self.activate_function(cor_max,self.th_cor)
            new_feature_score=np.clip(new_feature_score,a_min=None,a_max=new_feature_max_score)
            #save updated scores of selected features regarding current to-evaluate feature
            result_new_score[:,:,selected_feature_index]=temp_selected_feature_score
        #save scores of all to-evaluate features
        result_new_score[:,:,-1]=new_feature_score
        
        if self.performance_metric=='best performance':
            result_new_feature_index=np.argmin(np.sum(np.sum(result_new_score,axis=2),axis=0))
        elif self.performance_metric=='class balance':
            #shape is (n_class,1), tiled to (n_class,n_to_evaluate_feature), valudes are duplicated in cols to do matrix calculation
            avg_class_score_before=np.tile(np.mean(score_of_select_feature,axis=1).reshape(-1,1),(1,result_new_score.shape[1]))
            #shape is (n_class,n_to_evaluate_feature)
            avg_class_score_after=np.mean(result_new_score,axis=2)
            #only care about improvement(negative values)
            avg_class_score_improvement=np.clip(avg_class_score_after-avg_class_score_before,a_min=None,a_max=0)
            if np.min(avg_class_score_improvement)<0:
                result_new_feature_index=np.argmin(np.sum(avg_class_score_improvement,axis=0))
            #if no improvement, use best performance
            else:
                result_new_feature_index=np.argmin(np.sum(np.sum(result_new_score,axis=2),axis=0))
        return result_new_score[:,result_new_feature_index,:],new_feature_index_map[result_new_feature_index]
    
    def activate_function(self,value,threshold=0.5):    
        return np.clip(1./(1-threshold)*(value-threshold),a_min=0,a_max=1)
    
    def digitizeData(self,data):
        max_values=np.max(data,axis=0)
        min_values=np.min(data,axis=0)
        #use maimum of Freedman-Diaconis' rule and sturges to calculate bins, set minimum for data with few samples
        fd_binwidth=2*iqr(data,axis=0,interpolation='midpoint')*data.shape[0]**(-1/3)
        n_bins_fd=np.divide(max_values-min_values, fd_binwidth, out=np.zeros_like(fd_binwidth), where=fd_binwidth!=0)
        n_bins_sturges=math.log2(data.shape[0])+1
        nmi_bins=np.ceil(np.clip(np.maximum(n_bins_fd,n_bins_sturges),a_min=self.nmi_min_bins,a_max=data.shape[0])).astype(int)
        result=np.zeros(data.shape)
        for i in range(data.shape[1]):
            bins=np.linspace(min_values[i],max_values[i],nmi_bins[i])
            result[:,i]=np.digitize(data[:,i],bins)
        return result

    '''
    Parameters:
        data: (n_sample, n_feature) ndarray
        labels: (n_sample,) ndarray
        n_select: number of features to be selected
        threshold_cor: th for feature redudancy
        threshold_nmi: th for feature relevance
        ovo: one vs one, True for class-class pair comparison when estimating bayes error rate(BER), False is one vs all. class-all other classes comparison
        performance_metric: 'best performance' or 'class balance'. Best performance selects next feature which leads to best overall scores.
                            Class balance will select firstly feature with better(lower) BER than avg. BER of selected features(Liebig's law of the minimum). 
                            If there is no such feature, select next feature with best overall scores.
        min_kde_bandwidth: min. bandwidth for kde to estimate BER.
        kde_bins= bins used to do kde
        nmi_min_bins= min. bins to estimate normalized mutual information
        use_multiprocessing: currently always use multiprocessing
        num_multiprocessing: number of processings, default is number of cpu core
        verbose: whether do console output, always save log into file
    Returns:
        final_result: indice of selected features
        score_of_selected_features: scores of selected features, better feature has lower score
    '''
    def watermelon(self,data,labels,n_select=20,threshold_cor=0.5,threshold_nmi=0.3,ovo=True,performance_metric='class balance',min_kde_bandwidth=0.3,kde_bins=1000,nmi_min_bins=10,
                   use_multiprocessing=True,num_multiprocessing=None,verbose=True):
        '''create logger'''
        logger = logging.getLogger('Watermelon{}'.format(time.strftime('%Y%m%d-%H-%M-%S')))
        logger.setLevel(logging.DEBUG)
        # create file handler
        fh = logging.FileHandler('Watermelon feature selection result {}.log'.format(time.strftime('%Y%m%d-%H-%M-%S')))
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # create console handler if verbose=True
        if verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        '''init config'''
        self.th_cor=threshold_cor
        self.th_nmi=threshold_nmi
        self.ovo=ovo
        self.performance_metric=performance_metric
        self.min_kde_bw=min_kde_bandwidth
        self.kde_bins=kde_bins
        self.nmi_min_bins=nmi_min_bins
        if performance_metric != 'class balance' and performance_metric!= 'best performance':
            logger.error("performance_metric should be 'best performance' or 'class balance'")
            return
        '''data processing'''
        logger.debug('Data preprocessing...')
        #get number samples and features
        n_sample,n_feature=data.shape
        '''remove feature with 0 var'''
        zero_std_col=np.argwhere(np.var(data,axis=0)==0).flatten()[::-1]
        index_map=np.arange(n_feature)
        if len(zero_std_col) !=0:
            logger.debug('Removing features with zero variance...')
            #remove zero val col
            data=np.delete(data,zero_std_col,axis=1)
            #mapping of original index and new index
            for i in zero_std_col:
                index_map=np.delete(index_map,i)
            n_feature-=len(zero_std_col)
            
        #get classes
        labels=labels.astype('str')
        classes=np.unique(labels)
        n_class=len(classes)
        if n_sample<1 or n_feature<1 or n_class<2:
            logger.error('Input data not feasible')
            return
        digitized_data=self.digitizeData(data)
        logger.debug('Data preprocessing finished')
        
        '''start feature selection  '''  
        logger.debug('Feature selection in process...')
        logger.debug('Estimating Bayes error rate...')
        timer_start = time.time()
        
        '''calculate the score for each feature'''
        if use_multiprocessing:#TODO
            if num_multiprocessing is None:
                num_multiprocessing=os.cpu_count()
            #split data for multi processing
            data_splits=list(self.chunks(range(n_feature),math.ceil(n_feature/num_multiprocessing)))
            data_list=[data[:,split] for split in data_splits]
            #calculate BERs
            with Pool(processes=num_multiprocessing) as pool:
                results=pool.starmap(self.getErrorRateParallel,zip(data_list,repeat(labels),repeat(classes)))
            if self.ovo==True:
                n_class=comb(n_class,2,True)
            #cache feature scores
            score_of_rest_features=pd.DataFrame(np.zeros((n_class,n_feature),dtype='float'),columns=[str(i) for i in np.arange(n_feature)],dtype='float32')
            #update all feature scores according to BERs
            score_of_rest_features.iloc[:,:]=np.concatenate(results,axis=1)
        timer_BER=time.time()
        logger.debug('Estimation finisched. Time elapsed: {:.2f}s'.format(timer_BER-timer_start))
        
        '''select first feature'''
        #init selected feature score list
        score_of_selected_features=np.zeros((n_class,n_select),dtype='float')
        #get the init ranking after calculating the BERs
        result=[]
        #index of first n_select features with best BERs.
        sorted_index=np.argsort(np.sum(score_of_rest_features.values,axis=0))[:n_select]
        first_selection=sorted_index[0]
        #update scores
        result.append(first_selection)
        score_of_selected_features[:,0]=score_of_rest_features.loc[:,str(first_selection)].values
        score_of_rest_features.pop(str(first_selection))
        timer_first=time.time()
        logger.debug('1. feature selected. Time elapsed: {:.2f}s'.format(timer_first-timer_BER))
        
        '''calculate correlation and nmi'''
        logger.debug('Calculating spearman coefficients for all features...')
        #feature redundancy matrix according to spearman, nan will be treated as 1
        spearman_coe=np.abs(np.nan_to_num(stats.spearmanr(data)[0],nan=1))
        timer_spearman=time.time()
        logger.debug('Time elapsed: {:.2f}s'.format(timer_spearman-timer_first))
        #feature relevance matrix according to nmi
        nmi_coe=np.full((n_feature,n_feature),0.0001)
        logger.debug('Pre-calculating nmi for first {} features with best BERs...'.format(n_select))
        #mark which feature has calculated nmi
        nmi_calculated_index=np.full((n_feature,),-1,dtype='int')
        nmi_calculated_index[sorted_index]=[1]*len(sorted_index)
        #calculate nmi of fisrt n_select features to other features, do not calculate all the nmis due to time efficiency
        if use_multiprocessing:#TODO
            digitized_data_list=[digitized_data[:,split] for split in data_splits]
            with Pool(processes=num_multiprocessing) as pool:
                results=pool.starmap(self.getNmiParallel,zip(repeat(digitized_data[:,sorted_index]),digitized_data_list))
            nmi_coe[sorted_index,:]=np.concatenate(results,axis=1)
            nmi_coe[:,sorted_index]=nmi_coe[sorted_index,:].transpose()
        timer_nmi_first=time.time()
        logger.debug('Time elapsed: {:.2f}s'.format(timer_nmi_first-timer_spearman))
        # self.decay=1#TODO
        
        '''select 2. to n_select. feature'''
        for current_index in np.arange(1,n_select):
            timer_selection_start=time.time()
            #calculate new scores
            updated_score,updated_index=self.calculateFeatureScore(score_of_rest_features,score_of_selected_features[:,:len(result)],spearman_coe,nmi_coe,result)
            # update nmi
            if nmi_calculated_index[updated_index] != 1:
                with Pool() as pool:
                    results=pool.starmap(self.getNmiParallel,zip(repeat(digitized_data[:,updated_index].reshape(-1,1)),digitized_data_list))
                nmi_coe[:,updated_index]=nmi_coe[updated_index,:]=np.concatenate(results,axis=1)
                nmi_calculated_index[updated_index]=1
                
            result.append(updated_index)
            score_of_selected_features[:,:len(result)]=updated_score
            score_of_rest_features.pop(str(updated_index))

            timer_selection_end=time.time()
            logger.debug('{}. feature selected. Time elapsed: {:.2f}s'.format(current_index+1,timer_selection_end-timer_selection_start))
        # return result
        final_result=index_map[np.array(result)]
        timer_end = time.time()
        logger.debug('Feature selection finished. Time elapsed: {:.2f}s'.format(timer_end-timer_start))
        
        return final_result,score_of_selected_features
        
        

