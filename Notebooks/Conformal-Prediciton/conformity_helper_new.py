import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
class conform_helper():
    def calculate_metrics(self,targets, predictions):
        cm = confusion_matrix(targets, predictions)
        print(cm)
        f1 = f1_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        accuracy = accuracy_score(targets, predictions)
        return f1, precision, recall, accuracy,cm
    def coverage_modified_value(slef,prediction_sets,label):
        coverage_count=0
        non_coverage_count=0
        empty_set_count=0
        for i in range(0,prediction_sets.shape[0]):
            if prediction_sets[i][0]==prediction_sets[i][1]:
                empty_set_count+=1
            else:
                if prediction_sets[i][label[i]]==True:
                    coverage_count+=1
                else:
                    non_coverage_count+=1
        coverage_per=coverage_count/(prediction_sets.shape[0]-empty_set_count)
        non_coverage_per=non_coverage_count/(prediction_sets.shape[0]-empty_set_count)
        return coverage_count,non_coverage_count,coverage_per,non_coverage_per
    def get_conformity_result(self,TN,TP,FN,FP,probablity_data,y,y_pred,qhat):
        final_prediction_sets=probablity_data >= (1-qhat)
        print("this is final prediction sets")
        print(final_prediction_sets)
        conformity=[]
        for i in range(0,final_prediction_sets.shape[0]):
            if final_prediction_sets[i][0]==final_prediction_sets[i][1]:
                conformity.append('No-Conformity')
            else:
                conformity.append('Conform')
        df = pd.DataFrame({'Original':y , 'Prediction':y_pred,'Conformity':conformity})
        true_benign_conform=df.loc[(df['Conformity']=='Conform')&(df['Original']==0)&(df['Original']==df['Prediction'])]
        true_malware_conform=df.loc[(df['Conformity']=='Conform')&(df['Original']==1)&(df['Original']==df['Prediction'])]
        false_benign_conform=df.loc[(df['Conformity']=='Conform')&(df['Original']==0)&(df['Original']!=df['Prediction'])]
        false_malware_conform=df.loc[(df['Conformity']=='Conform')&(df['Original']==1)&(df['Original']!=df['Prediction'])]
        
        true_benign_No_conform=df.loc[(df['Conformity']=='No-Conformity')&(df['Original']==0)&(df['Original']==df['Prediction'])]
        true_malware_No_conform=df.loc[(df['Conformity']=='No-Conformity')&(df['Original']==1)&(df['Original']==df['Prediction'])]
        false_benign_No_conform=df.loc[(df['Conformity']=='No-Conformity')&(df['Original']==0)&(df['Original']!=df['Prediction'])]
        false_malware_No_conform=df.loc[(df['Conformity']=='No-Conformity')&(df['Original']==1)&(df['Original']!=df['Prediction'])]
        print(true_benign_conform.shape[0],true_malware_conform.shape[0],false_benign_conform.shape[0],false_malware_conform.shape[0],true_benign_No_conform.shape[0] \
        ,true_benign_No_conform.shape[0],true_malware_No_conform.shape[0],false_benign_No_conform.shape[0],false_malware_No_conform.shape[0])
        print("Correctly Predicted Benign:",TN,"Predicted Conform:",true_benign_conform.shape[0])
        print("Correctly Predicted Malware:",TP,"Predicted Conform:",true_malware_conform.shape[0])
        print("Original Benign but predicted Malware:",FP,"Predicted Conform:",false_benign_conform.shape[0])
        print("Original Malware but predicted Benign:",FN,"Predicted Conform:",false_malware_conform.shape[0])
        print("Correctly Predicted Benign:",TN,"Predicted Non-Conform:",true_benign_No_conform.shape[0])
        print("Correctly Predicted Malware:",TP,"Predicted Non-Conform:",true_malware_No_conform.shape[0])
        print("Original Benign but predicted Malware:",FP,"Predicted No-Conform:",false_benign_No_conform.shape[0])
        print("Original Malware but predicted Benign:",FN,"Predicted No-Conform:",false_malware_No_conform.shape[0])
        return true_benign_conform.shape[0],true_malware_conform.shape[0],false_benign_conform.shape[0],false_malware_conform.shape[0],true_benign_No_conform.shape[0],true_malware_No_conform.shape[0],false_benign_No_conform.shape[0],false_malware_No_conform.shape[0],df,final_prediction_sets
    
    def get_conformity_output(self,y_train_base_ucsb,probablity_train,train_pred_binary,y_val_base_ucsb,probability_val,val_pred_binary,ucsb_test_y,probability_test,test_pred_binary,q_hat):
        print("Train-Conformity")
        #ucsb get q hat lgb using validation
        train_f1, train_precision, train_recall, train_accuracy,train_cm = self.calculate_metrics(y_train_base_ucsb,train_pred_binary)
        train_TN, train_FP, train_FN, train_TP = train_cm[0][0], train_cm[0][1], train_cm[1][0], train_cm[1][1]
        print(train_TN, train_FP, train_FN, train_TP)
        train_tb_c,train_tm_c,train_fb_c,train_fm_c,train_tb_Nc,train_tm_Nc,train_fb_Nc,train_fm_Nc,train_df,lgb_train_final_prediction_sets=self.get_conformity_result(train_TN,train_TP, train_FN,train_FP,probablity_train,y_train_base_ucsb,train_pred_binary,q_hat)
        train_coverage_count,train_non_coverage_count,train_coverage_per,train_non_coverage_per=self.coverage_modified_value(lgb_train_final_prediction_sets,y_train_base_ucsb)
        print(f"Train| coverage count|{train_coverage_count}|Non coverage count|{train_non_coverage_count}| Coverage Percentage|{train_coverage_per}| Non Coverage Percentage|{train_non_coverage_per}")
        print("Calibration-Conformity")
        # calculate probability scores for LightGBM model predictions on validation set
        
        val_f1, val_precision, val_recall, val_accuracy, val_cm = self.calculate_metrics(y_val_base_ucsb,val_pred_binary)
        val_TN, val_FP, val_FN, val_TP = val_cm[0][0], val_cm[0][1], val_cm[1][0], val_cm[1][1]
        val_tb_c, val_tm_c, val_fb_c, val_fm_c, val_tb_Nc, val_tm_Nc, val_fb_Nc, val_fm_Nc, val_df, lgb_val_final_prediction_sets = self.get_conformity_result(val_TN, val_TP, val_FN, val_FP, probability_val, y_val_base_ucsb, val_pred_binary, q_hat)
        val_coverage_count, val_non_coverage_count, val_coverage_per, val_non_coverage_per = self.coverage_modified_value(lgb_val_final_prediction_sets, y_val_base_ucsb)
        print(f"Validation| Coverage count| {val_coverage_count} | Non-coverage count| {val_non_coverage_count} | Coverage percentage| {val_coverage_per} | Non-coverage percentage| {val_non_coverage_per}")
        print("Test-Conformity")
        # calculate probability scores for LightGBM model predictions on test set
        
        test_f1, test_precision, test_recall, test_accuracy, test_cm = self.calculate_metrics(ucsb_test_y,test_pred_binary)
        test_TN, test_FP, test_FN, test_TP = test_cm[0][0], test_cm[0][1], test_cm[1][0], test_cm[1][1]
        test_tb_c, test_tm_c, test_fb_c, test_fm_c, test_tb_Nc, test_tm_Nc, test_fb_Nc, test_fm_Nc, test_df, lgb_test_final_prediction_sets = self.get_conformity_result(test_TN, test_TP, test_FN, test_FP, probability_test, ucsb_test_y, test_pred_binary, q_hat)
        test_coverage_count, test_non_coverage_count, test_coverage_per, test_non_coverage_per = self.coverage_modified_value(lgb_test_final_prediction_sets, ucsb_test_y)
        print(f"Test| Coverage count| {test_coverage_count} | Non-coverage count| {test_non_coverage_count} | Coverage percentage| {test_coverage_per} | Non-coverage percentage| {test_non_coverage_per}")
        return train_tb_c,train_tm_c,train_fb_c,train_fm_c,train_tb_Nc,train_tm_Nc,train_fb_Nc,train_fm_Nc,train_TN, train_FP, train_FN, train_TP \
                ,val_tb_c, val_tm_c, val_fb_c, val_fm_c, val_tb_Nc, val_tm_Nc, val_fb_Nc, val_fm_Nc,val_TN, val_FP, val_FN, val_TP\
                ,test_tb_c, test_tm_c, test_fb_c, test_fm_c, test_tb_Nc, test_tm_Nc, test_fb_Nc, test_fm_Nc,val_TN, val_FP, val_FN, val_TP,\
                lgb_train_final_prediction_sets,lgb_val_final_prediction_sets,lgb_test_final_prediction_sets,train_df,val_df,test_df
