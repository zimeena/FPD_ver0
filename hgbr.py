# Ensemble method (gradient boosting) regression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.inspection import permutation_importance

from hp_optimization import *
import optuna
import lightgbm as lgbm
import joblib

from sklearn.metrics import r2_score, make_scorer
# from sklearn.utils import check_arrays
# stack overflow reference prd :https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
# stack overflow custom score refernces: https://stackoverflow.com/questions/51819803/custom-scoring-function-randomforestregressor
# Akshay's Last edited Copy
# Your custom scoring strategy
def prd(y_true, y_pred):
    # y_true, y_pred = check_arrays(y_true, y_pred)
    print(np.median(np.abs((y_true - y_pred) / y_true)))
    return np.mean(np.abs((y_true - y_pred) / y_true))

# Wrapping it in make_scorer to able to use in RandomizedSearch
my_scorer = make_scorer(prd)


def hgbr(train_df, validation_df,test_df,experiment, scenario,tune_hyperparameters,metric_to_optimize, use_lightgbm,start_region, end_region, num_hyperparameter_search_candidates):
    basin_HGBR_models = []
    # test_example_df = pd.read_csv('huc_15_09404450_test_data_sample.csv')
    # test_example_df = test_example_df[['P_Trig_max', 'P_Trig_mean', 'API', 'pet_mean', 'frac_forest', 'soil_porosity', 'P_Trig_Temp_Max_av', 'Normalized_Peak']]

    results_df = pd.DataFrame(list(validation_df.columns).append('Predicted_Normalized_Peak'))
    

    top_list = []
    top_mean_list = []
    top_sd_list = []
    importance_df = pd.DataFrame(columns=train_df.drop(columns=['Normalized_Peak','label', 'huc_02']).columns)
    row_list = []



    for i in range(start_region,end_region):
        # get the dataset for the specific basin
        train_df_basin = train_df[train_df['huc_02'] == i] 
        # print(train_df_basin['Normalized_Peak'])
        validation_df_basin = validation_df[validation_df['huc_02'] == i]
        test_df_basin = test_df[test_df['huc_02'] == i]

        train_df_basin = train_df_basin.dropna()
        validation_df_basin = validation_df_basin.dropna()
        test_df_basin = test_df_basin.dropna()

        # find threshold for the basin
        # training threshold
        training_threshold = np.percentile(train_df_basin['Normalized_Peak'].values, 75)
        # validation threshold
        validation_threshold = np.percentile(validation_df_basin['Normalized_Peak'].values, 75)

        overall_peak_75_threshold = np.average([training_threshold, validation_threshold])

        new_df = train_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        Y_train_basin_all_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_train_basin_all_flows = new_df.values
        columns = new_df.columns

        new_df = validation_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        Y_validation_basin_all_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_validation_basin_all_flows = new_df.values

        new_df = test_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        Y_test_basin_all_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_test_basin_all_flows = new_df.values

        new_df = train_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] < overall_peak_75_threshold]

        Y_train_basin_low_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_train_basin_low_flows = new_df.values
        columns = new_df.columns

        new_df = validation_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] < overall_peak_75_threshold]
        validation_df_basin_low_flows = validation_df_basin[validation_df_basin['Normalized_Peak'] < overall_peak_75_threshold]
        Y_validation_basin_low_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_validation_basin_low_flows = new_df.values


        new_df = test_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] < overall_peak_75_threshold]
        test_df_basin_low_flows = test_df_basin[test_df_basin['Normalized_Peak'] < overall_peak_75_threshold]
        Y_test_basin_low_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_test_basin_low_flows = new_df.values

        new_df = train_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] >= overall_peak_75_threshold]
        Y_train_basin_high_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_train_basin_high_flows = new_df.values
        columns = new_df.columns

        new_df = validation_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] >= overall_peak_75_threshold]
        validation_df_basin_high_flows = validation_df_basin[validation_df_basin['Normalized_Peak'] >= overall_peak_75_threshold]
        Y_validation_basin_high_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_validation_basin_high_flows = new_df.values


        new_df = test_df_basin.drop('CatchmentID',1)
        new_df = new_df.dropna()
        new_df = new_df[new_df['Normalized_Peak'] >= overall_peak_75_threshold]
        test_df_basin_high_flows = test_df_basin[test_df_basin['Normalized_Peak'] >= overall_peak_75_threshold]
        Y_test_basin_high_flows = new_df['Normalized_Peak'].values # unit of Peak is ft^-3
        new_df = new_df.drop(columns=['Normalized_Peak','label', 'huc_02'])
        X_test_basin_high_flows = new_df.values

        def hyper_parameter_fitting(huc):

            def objective(trial):
                max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 60, 1)
                #trial.suggest_categorical("max_leaf_nodes",[2,3,4,5,6,7,8,9,10,20,30,40,50])
                max_iter = trial.suggest_int("max_iter", 20, 300, 10)
                #trial.suggest_categorical("max_iter", [75, 80, 90, 100, 120, 140, 150])
                l2_regularization = trial.suggest_float("l2_regularization", 0, 2000)
                #trial.suggest_categorical("l2_regularization", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-1, log=True)
                #trial.suggest_categorical("learning_rate", [0.01, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5])
            
                if use_lightgbm:
                    params = {
                                'task': 'train', 
                                'boosting': 'gbdt',
                                'objective': metric_to_optimize,
                                'num_leaves': max_leaf_nodes,
                                'num_iterations':max_iter,
                                'learning_rate': learning_rate,
                                'lambda_l2':l2_regularization, 
                                # 'metric': {'l2','l1'},
                                'verbose': 1
                            }
                    lgb = lgbm.LGBMRegressor(**params)
                    
                    # HGBR_object = lgb.train(train_set=training_data,valid_sets = validation_data, 
                    #                             params=params)
                    HGBR_object = lgb.fit(X_train_basin_all_flows,Y_train_basin_all_flows,
                                            eval_set =(X_validation_basin_all_flows, Y_validation_basin_all_flows), 
                                                )

                    
                else:
                    validation_fraction = trial.suggest_categorical("validation_fraction",  [0.1, 0.15, 0.2])
                
                    HGBR_object = HistGradientBoostingRegressor(scoring=my_scorer,early_stopping=True,max_leaf_nodes = max_leaf_nodes, verbose=0,max_iter=max_iter, 
                                                                validation_fraction=validation_fraction, l2_regularization=l2_regularization, learning_rate=learning_rate)
                    HGBR_object.fit(X_train_basin_all_flows, Y_train_basin_all_flows)
                
                Y_pred_test = HGBR_object.predict(X_test_basin_all_flows)
                Y_pred_validation = HGBR_object.predict(X_validation_basin_all_flows)
                
                val_rmse = np.sqrt(mean_squared_error(Y_validation_basin_all_flows, Y_pred_validation))
                val_prd = prd(Y_validation_basin_all_flows, Y_pred_validation)
                
                if metric_to_optimize == "mape":
                    return val_prd
                elif metric_to_optimize == "mse":
                    return val_rmse
                else:
                    raise Exception("Metric "+metric_to_optimize+" has not been implemented")
                
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=num_hyperparameter_search_candidates)
            print(study.best_params)
            results_csv = pd.read_csv('results.csv')
            results_dict = {'huc': huc, 'scenario':scenario, 'experiment': experiment,  'best_params':study.best_params}

            results_csv = results_csv.append(results_dict, ignore_index=True)
            results_csv.to_csv('results.csv',mode='w',index=False)
            

        print('Number of basins trained: ', i)

        if tune_hyperparameters:
            hyper_parameter_fitting(i)
        else:
            if not use_lightgbm:
                HGBR_object = HistGradientBoostingRegressor(scoring=my_scorer,early_stopping=True,max_leaf_nodes = best_param_dict[experiment][i]['max_leaf_nodes'],
                                                        validation_fraction = best_param_dict[experiment][i]['validation_fraction'],
                                                        max_iter = best_param_dict[experiment][i]['max_iter'],
                                                        l2_regularization=best_param_dict[experiment][i]['l2_regularization'],
                                                        learning_rate=best_param_dict[experiment][i]['learning_rate'],verbose=0)
        
        
            
        #Train_IMG_CONUS_01_Valid_IMG_UK_01
        
        if not tune_hyperparameters:
            if scenario == "Train_LOCAL_CONUS_07_Test_LOCAL_CONUS_07":

                if use_lightgbm:
                    params = {
                                'task': 'train', 
                                'boosting': 'gbdt',
                                'objective': metric_to_optimize,
                                'num_leaves':  best_param_dict[experiment][i]['max_leaf_nodes'],
                                'num_iterations': best_param_dict[experiment][i]['max_iter'],
                                'learning_rate': best_param_dict[experiment][i]['learning_rate'],
                                'lambda_l2':best_param_dict[experiment][i]['l2_regularization'], 
                                'verbose': 1
                            }
                    lgb = lgbm.LGBMRegressor(**params)
                    
                    # HGBR_object = lgb.train(train_set=training_data,valid_sets = validation_data, 
                    #                             params=params)
                    HGBR_object = lgb.fit(X_train_basin_all_flows,Y_train_basin_all_flows,
                                            eval_set =(X_validation_basin_all_flows, Y_validation_basin_all_flows), 
                                                )
                else:
                    HGBR_object.fit(X_train_basin_all_flows, Y_train_basin_all_flows)
                    
                Y_pred_test = HGBR_object.predict(X_test_basin_all_flows)
                Y_pred_val = HGBR_object.predict(X_validation_basin_all_flows)
                
                print("Test RMSE:", np.sqrt(mean_squared_error(Y_test_basin_all_flows, Y_pred_test)))
                print("Val RMSE:", np.sqrt(mean_squared_error(Y_validation_basin_all_flows, Y_pred_val)))
                print("Test PRD:",prd(Y_test_basin_all_flows, Y_pred_test))
                print("Validation PRD:",prd(Y_validation_basin_all_flows, Y_pred_val))
                test_df_basin['Predicted_Normalized_Peak'] = Y_pred_test
                results_df = results_df.append(test_df_basin)
                basin_HGBR_models.append(HGBR_object)
            else:
                if use_lightgbm:
                    
                    params = {
                                'task': 'train', 
                                'boosting': 'gbdt',
                                'objective': metric_to_optimize,
                                'num_leaves':  best_param_dict[experiment][i]['max_leaf_nodes'],
                                'num_iterations': best_param_dict[experiment][i]['max_iter'],
                                'learning_rate': best_param_dict[experiment][i]['learning_rate'],
                                'lambda_l2':best_param_dict[experiment][i]['l2_regularization'], 
                                'metric': {'l2','l1'},
                                'verbose': 1
                            }
                    lgb = lgbm.LGBMRegressor(**params)
                    
                    # HGBR_object = lgb.train(train_set=training_data,valid_sets = validation_data, 
                    #                             params=params)
                    HGBR_object = lgb.fit(eval('X_train_basin_'+str(scenario)+'_flows'),eval('Y_train_basin_'+str(scenario)+'_flows'),
                                            eval_set =(eval('X_validation_basin_'+str(scenario)+'_flows'), eval('Y_validation_basin_'+str(scenario)+'_flows')), 
                                        )
                    
                    # HGBR_object = lgb.train(train_set=training_data,valid_sets = validation_data, 
                    #                             params=params)
                else:
                    HGBR_object.fit(eval('X_train_basin_'+str(scenario)+'_flows'), eval('Y_train_basin_'+str(scenario)+'_flows') )
                
                Y_pred_test = HGBR_object.predict(eval('X_test_basin_'+str(scenario)+'_flows'))
                eval('test_df_basin_'+str(scenario)+'_flows')['Predicted_Normalized_Peak'] = Y_pred_test
                print("PRD:",prd(eval('Y_test_basin_'+str(scenario)+'_flows'),Y_pred_test))
                results_df = results_df.append(eval('test_df_basin_'+str(scenario)+'_flows'))
                basin_HGBR_models.append(HGBR_object)
                print('Number of basins trained: ', i, end='\r')


            r = permutation_importance(HGBR_object, X_validation_basin_all_flows, Y_validation_basin_all_flows,n_repeats=5,random_state=0)
            top_five = columns[r.importances_mean.argsort()[::-1]]
            top_mean = r.importances_mean[r.importances_mean.argsort()[::-1]]
            top_std = r.importances_std[r.importances_mean.argsort()[::-1]]
            row_dict = {}
            for i in r.importances_mean.argsort()[::-1]:      
                row_dict[columns[i]] = r.importances_mean[i]
            row_list.append(row_dict)
            top_list.append(top_five)
            top_mean_list.append(top_mean)
            top_sd_list.append(top_std)
        
    importance_df = pd.DataFrame(row_list)
    importance_df.to_csv(experiment+"_HGBR_feature_importance_"+str(scenario)+"_flows.csv")
    if not tune_hyperparameters:
        results_df = results_df[['CatchmentID','Normalized_Peak','Predicted_Normalized_Peak','huc_02']]
        print("Number of samples ",len(results_df))
        results_df.to_csv(experiment+'_HGBR_test_results_'+str(scenario)+'_flows.csv',index=False)
