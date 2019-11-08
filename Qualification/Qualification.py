import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool as ProcessPool

import warnings 
warnings.filterwarnings('ignore')


df_user_testA = pd.read_csv('../data/ECommAI_EUIR_round1_testA_20190701/user.csv',header=None)
df_user_testA.columns = ['userID','gender','age','purchaseLevel']
df_item_testA = pd.read_csv('../data/ECommAI_EUIR_round1_testA_20190701/item.csv',header=None)
df_item_testA.columns = ['itemID','categoryID','shopID','brandID']
df_log_testA = pd.read_csv('../data/ECommAI_EUIR_round1_testA_20190701/user_behavior.csv',header=None)
df_log_testA.columns = ['userID','itemID','behavior','timestap']

df_user_testB = pd.read_csv('../data/ECommAI_EUIR_round1_testB_20190809/user.csv',header=None)
df_user_testB.columns = ['userID','gender','age','purchaseLevel']
df_item_testB = pd.read_csv('../data/ECommAI_EUIR_round1_testB_20190809/item.csv',header=None)
df_item_testB.columns = ['itemID','categoryID','shopID','brandID']
df_log_testB = pd.read_csv('../data/ECommAI_EUIR_round1_testB_20190809/user_behavior.csv',header=None)
df_log_testB.columns = ['userID','itemID','behavior','timestap']




df_log_testA['date'] = df_log_testA['timestap'].apply(lambda x : datetime.datetime(2019,7,5) + datetime.timedelta(seconds=x))
df_log_testA['day'] = df_log_testA['date'].dt.day
df_log_testA['weekday'] = df_log_testA['date'].dt.weekday + 1

df_log_testB['date'] = df_log_testB['timestap'].apply(lambda x : datetime.datetime(2019,7,5) + datetime.timedelta(seconds=x))
df_log_testB['day'] = df_log_testB['date'].dt.day
df_log_testB['weekday'] = df_log_testB['date'].dt.weekday + 1



df_log = pd.concat([df_log_testA,df_log_testB])
df_log.reset_index(inplace=True,drop=True)


df_log_selected = df_log.copy()


df_log_selected.loc[df_log_selected['behavior']=='pv','behavior'] = 1
df_log_selected.loc[df_log_selected['behavior']=='fav','behavior'] = 2
df_log_selected.loc[df_log_selected['behavior']=='cart','behavior'] = 3
df_log_selected.loc[df_log_selected['behavior']=='buy','behavior'] = 4

df_log_selected['hour'] = df_log_selected['date'].dt.hour
df_log_selected['day_hour'] = df_log_selected['day'] + df_log_selected['hour']/float(24)
df_log_selected['behavior'] = (1 - (21-df_log_selected['day_hour']+1)/(21-5+1)) * df_log_selected['behavior']




df_user = pd.concat([df_user_testA,df_user_testB])
df_user.reset_index(inplace=True,drop=True)

df_log_selected = pd.merge(df_log_selected,df_user)
df_log_selected_male = df_log_selected[ df_log_selected['gender']==0 ]
df_log_selected_female = df_log_selected[ df_log_selected['gender']==1 ]



item_statistc_male_w = df_log_selected_male.groupby(['itemID'])[['behavior']].sum()
item_statistc_male_w.reset_index(inplace=True)
item_statistc_male_w.columns = ['itemID','itemCount_male_w']



item_statistc_female_w = df_log_selected_female.groupby(['itemID'])[['behavior']].sum()
item_statistc_female_w.reset_index(inplace=True)
item_statistc_female_w.columns = ['itemID','itemCount_female_w']



df_item = pd.concat([df_item_testA,df_item_testB])
df_item.drop_duplicates(inplace=True)
df_item.reset_index(drop=True,inplace=True)


df_item = pd.merge(df_item,item_statistc_female_w,how='left')
df_item = pd.merge(df_item,item_statistc_male_w,how='left')

df_item.loc[np.isnan(df_item['itemCount_male_w']),'itemCount_male_w' ] = 0
df_item.loc[np.isnan(df_item['itemCount_female_w']),'itemCount_female_w' ] = 0



df_item_male_sorted = df_item.sort_values(by=['itemCount_male_w'],ascending=False)
df_item_male_sorted.reset_index(drop=True,inplace=True)
item_male_w_Top500 = list( df_item_male_sorted.loc[:499,'itemID'] )



df_item_female_sorted = df_item.sort_values(by=['itemCount_female_w'],ascending=False)
df_item_female_sorted.reset_index(drop=True,inplace=True)
item_female_w_Top500 = list( df_item_female_sorted.loc[:499,'itemID'] )



item_to_category_dict = dict()
for row in df_item.values:
    item_to_category_dict[row[0]] = row[1]


historicalDict = {}
for each_user in df_user_testB['userID']:
    historicalDict[each_user]= set(df_log_selected.loc[df_log_selected['userID']==each_user,'itemID'])


his_cat_Dict = {}

for each_user in df_user_testB['userID']:
    cat_list = []
    for item_t in historicalDict[each_user]:
        cat_list.append(item_to_category_dict[item_t])
	
    his_cat_Dict[each_user] = set(cat_list)







def process(each_user):

    gender_tmp = int(df_user_testB.loc[df_user_testB['userID']==each_user,'gender'])
    
    df_tmp = df_log_selected[df_log_selected['userID']==each_user]
    df_tmp.reset_index(inplace=True,drop=True)
    
    cat_his_tmp = his_cat_Dict[each_user]
    
    itemListTmp = []
    
    if len(df_tmp) > 0:
        
        item_sta = df_tmp.groupby(['itemID'])['behavior'].sum()
        item_sta = item_sta.reset_index()
        item_sta_sorted = item_sta.sort_values(by=['behavior'],ascending=False)
        item_sta_sorted.reset_index(inplace=True,drop=True)
        
        itemListTmp = itemListTmp + list(item_sta_sorted.loc[:27,'itemID'])
        

        
        if len(itemListTmp) < 50:
               
            if gender_tmp == 0:

                    for item_candidate in item_male_w_Top500:
                        if item_candidate not in itemListTmp and item_to_category_dict[item_candidate] not in cat_his_tmp:
                            itemListTmp.append(item_candidate)
                        
                        if len(itemListTmp) == 50:
                            break
            else:

                    for item_candidate in item_female_w_Top500:
                        if item_candidate not in itemListTmp and item_to_category_dict[item_candidate] not in cat_his_tmp:
                            itemListTmp.append(item_candidate)
                        
                        if len(itemListTmp) == 50:
                            break            
        
    else:
        
        if gender_tmp == 0:
            
                for item_candidate in item_male_w_Top500:
                    if item_candidate not in itemListTmp and item_to_category_dict[item_candidate] not in cat_his_tmp:
                        itemListTmp.append(item_candidate)
                    
                    if len(itemListTmp) == 50:
                        break

        else:
                for item_candidate in item_female_w_Top500:
                    if item_candidate not in itemListTmp and item_to_category_dict[item_candidate] not in cat_his_tmp:
                        itemListTmp.append(item_candidate)
                    
                    if len(itemListTmp) == 50:
                        break
	
    
    return (each_user,set(itemListTmp))

pool = ProcessPool(8)
res = pool.map(process, df_user_testB['userID'])
pool.close()
pool.join()


file = open('../prediction_result/result.csv','w')


for element in res:
    strTmp = str(element[0]) + ',' + ','.join(map(lambda x:str(x), list(element[1]) ))
    file.write(strTmp+'\n')

file.close()



