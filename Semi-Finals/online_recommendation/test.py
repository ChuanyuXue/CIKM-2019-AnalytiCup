import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
import lightgbm as lgb

start_time = time.time()
# Note! 别忘了改线上路径
path = '/tcdata/'
#path = './testA/'

object_path = '/competition/'
#object_path = './'

targetday = 16

static_features_path = ''

static_features_files = [
'brand_count.csv',
'brand_sum.csv',
'category_count.csv',
'category_sum.csv',
'itemID_count.csv',
'itemID_sum.csv',
'shop_count.csv',
'shop_sum.csv',
'category_lower.csv',
'item_rank.csv',
'category_higher.csv',
'itemID_higher.csv',
]

model_name = 'model0924_base.file'

time_features_files = [
'itemID_last_time_online.csv',
'brand_last_time_online.csv',
'shop_last_time_online.csv'
]

double_features_files = [
'item_to_ability_count_online.csv',
'item_to_sex_count_online.csv',
'item_to_age_count_online.csv',
]


tempory_flie_path = ''


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """ 
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    
    return df

def load_uandi(path):
    user = reduce_mem_usage(pd.read_csv(path + 'user.csv',header=None, engine='c'))
    item = reduce_mem_usage(pd.read_csv(path + 'item.csv',header=None, engine='c'))
    item.columns = ['itemID','category','shop','brand']
    user.columns = ['userID','sex','age','ability']

    return user, item



def load_data(path):
    '''
    input: the directory of original dataset
    output: user, item, data(tapped with item&user attributes and behavior features)
    '''

    user = reduce_mem_usage(pd.read_csv(path + 'user.csv',header=None, engine='c'))
    item = reduce_mem_usage(pd.read_csv(path + 'item.csv',header=None, engine='c'))
    data = pd.read_csv(path + 'user_behavior.csv',header=None, engine='c')

    data.columns = ['userID','itemID','behavior','timestamp']
    data['day'] = data['timestamp'] // 86400
    data['hour'] = data['timestamp'] // 3600 % 24
    
    ## 生成behavior的onehot
    for i in ['pv','fav','cart','buy']:
        data[i] = 0
        data.loc[data['behavior'] == i, i] = 1

    ## 生成behavior的加权
    
    data['day_hour'] = data['day'] + data['hour'] / float(24)
    data.loc[data['behavior']=='pv','behavior'] = 1
    data.loc[data['behavior']=='fav','behavior'] = 2
    data.loc[data['behavior']=='cart','behavior'] = 3
    data.loc[data['behavior']=='buy','behavior'] = 1
    max_day = max(data['day'])
    min_day = min(data['day'])
    data['behavior'] = (1 - (max_day-data['day_hour']+2)/(max_day-min_day+2)) * data['behavior'] 

    item.columns = ['itemID','category','shop','brand']
    user.columns = ['userID','sex','age','ability']
    
    data = reduce_mem_usage(data)

    data = pd.merge(left=data, right=item, on='itemID',how='left', sort=False)
    data = pd.merge(left=data, right=user, on='userID',how='left', sort=False)

    return user, item, data


def get_unique_inorder(x, k=50):
    '''
    input: Iterable x, Int k
    return: Iterable x(keep first 50 unique elements)
    '''

    result = []
    flag = set()
    for i in x:
        if i[0] not in flag:
            result.append(i)
            flag.add(i[0])
        if len(flag) > k:
            break
    return result


def get_recall_list(train, targetDay, tempory_flie_path = './', k=300):
    '''
    input: DataFrame train(data used for extracting recall, )
    '''

    train_logs = dict()
    f = open(tempory_flie_path + 'upward_map.txt','r')
    upward_map = f.read()
    upward_map = eval(upward_map)
    f.close()
    
    f = open(tempory_flie_path + 'downward_map.txt','r')
    downward_map = f.read()
    downward_map = eval(downward_map)
    f.close()
    

    f = open(tempory_flie_path + 'item_Apriori.txt','r')
    tmp = f.read()
    item_dict = eval(tmp)
    f.close()
    
    if targetDay > max(train['day']):
        for row in train[['userID','itemID','behavior']].values:
            train_logs.setdefault(row[0], dict())
            if row[1] in upward_map:
                train_logs[row[0]].setdefault(upward_map[row[1]],0)
                train_logs[row[0]][upward_map[row[1]]] = max(train_logs[row[0]][upward_map[row[1]]],row[2])
    else:
        user_List_test = set(train.loc[train['day']==targetDay,'userID'])
        train = train[train['day'] < targetDay]
        
        for row in train[['userID','itemID','behavior']].values:
            if row[0] in user_List_test:
                train_logs.setdefault(row[0], dict())
                if row[1] in upward_map:
                    train_logs[row[0]].setdefault(upward_map[row[1]],0)
                    train_logs[row[0]][upward_map[row[1]]] = max(train_logs[row[0]][upward_map[row[1]]],row[2])

    for each_user in train_logs:
        sum_value = sum(train_logs[each_user].values())
        if sum_value > 0:
            for each_item in train_logs[each_user]:
                train_logs[each_user][each_item] /= sum_value            

    result_logs = dict()    
    for u in train_logs:
        result_logs.setdefault(u, list())
        for i in set(train_logs[u].keys()):
            if i in item_dict:
                tmp_list = [ (x[0], train_logs[u][i]*x[1]) for x in item_dict[i]]
                result_logs[u] += tmp_list
            
    for u in result_logs:
        result_logs[u] = get_unique_inorder([(downward_map[x[0]], x[1]) for x in sorted(result_logs[u], key=lambda x:x[1], reverse=True)
                          if x[0] not in train_logs[u]], k=k)  
    
    return result_logs


def generate_pairs(recall):
    result = []
    for u in recall:
        for i in recall[u]:
            result.append([u,i[0],i[1]])
    return result

def reshape_recall_to_dataframe(recall):
    result = generate_pairs(recall)
    result = pd.DataFrame(result)
    result.columns = ['userID','itemID','apriori']
    return result

def recall(dict1, dict2, train_dict):
    '''
    dict1 是真值 dict2 是预测值.
    '''
    
    result = 0
    count = 0
    for i in dict1:
        if i in dict2 and i in train_dict:
            new_item = set()
    
            for k in dict1[i]:
                if k not in train_dict[i]:
                    new_item.add(k)
            if new_item:
                result += len(new_item & set(dict2[i])) / len(new_item)
                count += 1
            
    if count == 0:
        return 0
    else:
        return result / count

def generate_online_features(data):
    online_features = []
    for count_feature in ['category','shop','brand']:
        online_features.append(data[['behavior','userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
            {'behavior': 'count'}).rename(columns={'behavior':'user_to_'
                                                   + count_feature + '_count'}))
    for count_feature in ['category','shop','brand']:
        online_features.append(data[['behavior','userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
            {'behavior': 'sum'}).rename(columns={'behavior':'user_to_' 
                                                 + count_feature + '_sum'}))
    for count_feature in ['category','shop','brand']:
        for behavior_type in ['pv','buy']:
            online_features.append(data[[behavior_type,'userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
                {behavior_type: 'sum'}).rename(columns={behavior_type:'user_to_'
                                                       + count_feature + '_count_' + behavior_type}))

    return online_features

def generate_yestday_features(data, targetday):
    yestday_features = []
    yestday = data[data['day'] == targetday - 1]
    
    for count_feature in ['category','shop','brand']:
        yestday_features.append(yestday[['behavior','userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
            {'behavior': 'count'}).rename(columns={'behavior':'user_to_'
                                                   + count_feature + '_count_yestday'}))

    for count_feature in ['category','shop','brand']:
        for behavior_type in ['pv','buy']:
            yestday_features.append(yestday[[behavior_type,'userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
                {behavior_type: 'sum'}).rename(columns={behavior_type:'user_to_'
                                                       + count_feature + '_count_'+behavior_type+'_yestday'}))
    return yestday_features

def generate_5days_features(data, targetday):
    a5days = data[(data['day'] > targetday - 1 - 5) & (data['day'] < targetday - 1)]
    five_days_features = []
    
    for count_feature in ['category','shop','brand']:
        five_days_features.append(a5days[['behavior','userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
            {'behavior': 'count'}).rename(columns={'behavior':'user_to_'
                                                   + count_feature + '_count_5days'}))

    for count_feature in ['category','shop','brand']:
        for behavior_type in ['pv','fav','cart','buy']:
            five_days_features.append(a5days[[behavior_type,'userID',count_feature]].groupby(['userID', count_feature], as_index=False).agg(
                {behavior_type: 'sum'}).rename(columns={behavior_type:'user_to_'
                                                       + count_feature + '_count_' + behavior_type+'_5days'}))
    return five_days_features
        
def generate_lasttime_features(data, targetday):
    dynamic_time_features = []
    test = data[data['day'] < targetday]
    start_timestamp  = max(test['timestamp'])
    test['lasttime'] = start_timestamp - test['timestamp']
    
    for dynamic_time_feature in ['shop', 'category','brand']:
        dynamic_time_features.append(test[['lasttime','userID',dynamic_time_feature,'day']].groupby(['userID',dynamic_time_feature], as_index=False).agg({'lasttime': 'min', 'day':'max'}).rename(columns={'lasttime': 'user_to_'
                                                       + dynamic_time_feature + '_lasttime', 'day':'user_to_'+ dynamic_time_feature + '_lastday'}))
    return dynamic_time_features



#--------------------------------------------------------------------------
# 流程开始

user, item, data = load_data(path)
user['age'] = user['age'] // 10
data['age'] = data['age'] // 10

test_recall_logs = get_recall_list(data, targetDay=targetday, tempory_flie_path=tempory_flie_path, k=325)

test_recall = reshape_recall_to_dataframe(test_recall_logs)
test_recall = pd.merge(left=test_recall, right=user, on='userID',how='left', sort=False)
test_recall = pd.merge(left=test_recall, right=item, on='itemID',how='left', sort=False)

recall_time = time.time()
print(str((recall_time - start_time) // 60) + ' is cost in recall')

# Concat time features
time_features = []
for f in time_features_files:
    time_features.append(reduce_mem_usage(pd.read_csv(f, engine='c')))

for f in time_features:
    test_recall = pd.merge(left=test_recall, right=f, on=f.columns[0], how='left', sort=False)
time_features = []

# Concat static features
static_features = []
for f in static_features_files:
    static_features.append(reduce_mem_usage(pd.read_csv(static_features_path + f, engine='c')))

for f in static_features:
    test_recall = pd.merge(left=test_recall, right=f, on=f.columns[0], how='left', sort=False)
static_features = []

# Concat double features

double_features = []
for f in double_features_files:
    double_features.append(reduce_mem_usage(pd.read_csv(static_features_path + f, engine='c')))

for f in double_features:
    test_recall = pd.merge(left=test_recall, right=f, on=list(f.columns[0: 2]), how='left', sort=False)
double_features = []


# Concat dynamic features
dynamic_features = generate_online_features(data)

for f in dynamic_features:
    test_recall = pd.merge(left=test_recall, right=f, on=list(f.columns[0: 2]), how='left', sort=False)
dynamic_features = []

yestday_features = generate_yestday_features(data, targetday = targetday)
for f in yestday_features:
    test_recall = pd.merge(left=test_recall, right=f, on=list(f.columns[0: 2]), how='left', sort=False)
yestday_features = []

fiveday_features = generate_5days_features(data, targetday = targetday)
for f in fiveday_features:
    test_recall = pd.merge(left=test_recall, right=f, on=list(f.columns[0: 2]), how='left', sort=False)
fiveday_features = []

last_time_features = generate_lasttime_features(data, targetday = targetday)
for f in last_time_features:
    test_recall = pd.merge(left=test_recall, right=f, on=list(f.columns[0: 2]), how='left', sort=False)
last_time_features = []

feature_time = time.time()
print(str((feature_time - recall_time) // 60) + ' is cost in feature')

model = CatBoostClassifier()

model = model.load_model(model_name)

features = [x for x in test_recall.columns if x not in ['itemID','userID','category','shop','brand','label']]

test_recall['label'] = model.predict_proba(test_recall[features])[:,1]

features = ['apriori',
 'user_to_category_count_pv_5days',
 'user_to_category_count_buy',
 'itemID_sum',
 'itemID_count',
 'user_to_category_count_pv',
 'user_to_category_count_5days',
 'itemIDlast_time',
 'user_to_category_count_pv_yestday',
 'shop_count',
 'category_count',
 'user_to_category_count_yestday',
 'user_to_category_sum',
 'user_to_brand_count_pv',
 'user_to_shop_sum',
 'user_to_shop_count_pv',
 'age',
 'shop_sum',
 'category_sum',
 'brand_count',
 'user_to_shop_count',
 'user_to_brand_count_pv_5days',
 'rank',
 'rank_percent',
 'ability',
 'user_to_age_count',
 'user_to_sex_count',
 'user_to_shop_count_buy',
 'user_to_shop_count_pv_5days',
 'user_to_shop_count_pv_yestday',
 'user_to_shop_lasttime',
 'user_to_category_count',
 'user_to_category_lasttime',
 'category_median',
 'category_skew',
 'category_std'
]

model_lgb = lgb.Booster(model_file='lgb_0924_1652')
test_recall['label_lgb'] = model_lgb.predict(test_recall[features])


#0.045965784783714
test_recall['ensemble'] = 10 / ( 5/test_recall['label_lgb'] + 5/test_recall['label'])

#0.045943749548558184
test_recall['ensemble_power'] = np.power( test_recall['label_lgb']**4.8 * test_recall['label']**5.2 , 1/10)

#0.045996441844155474
test_recall['ensemble_final'] = test_recall['ensemble']*0.5 + test_recall['ensemble_power'] * 0.5

model_time = time.time()
print(str((model_time - feature_time) // 60) + ' is cost in model')

train_logs = dict()
train_ = data
for row in train_[['userID','itemID']].values:
    train_logs.setdefault(row[0], [])
    train_logs[row[0]].append(row[1])

result_logs = dict()
test_recall = test_recall.sort_values('ensemble_final', ascending=False).reset_index(drop=True)
for row in test_recall[['userID','itemID']].values:
    result_logs.setdefault(row[0], [])
    if len(result_logs[row[0]]) < 50:
        result_logs[row[0]].append(row[1])

temp = data.groupby(['itemID'], as_index=False).count()[['itemID','userID']]
hot_items = list(temp.sort_values('userID', ascending=False).reset_index(drop=True)['itemID'][:100])

rec_dict = dict()
for u in set(data['userID']):
    if u in result_logs:
        lenth = len(result_logs[u])
        if lenth < 50:
            rec_dict[u] = result_logs[u] + [x for x in hot_items if x not in result_logs[u] and x not in train_logs[u]][:50 - lenth]
        else:
            rec_dict[u] = result_logs[u]
    else:
        rec_dict[u] = [x for x in hot_items][:50]


# Note! 别忘了改线上路径
file = open(object_path + 'result.csv','w')
for element in rec_dict:
    strTmp = str(int(element)) + ',' + ','.join(map(lambda x:str(int(x)), list(rec_dict[element] ) ))
    file.write(strTmp+'\n')
file.close()

final_time = time.time()
print(str((final_time - start_time) // 60) + ' is cost in whole process')

#--------------------------------------------------------------
# 下面用于线下测试

# train_logs = dict()
# train = data[data['day'] < 15]
# for row in train[['userID','itemID']].values:
#     train_logs.setdefault(row[0], [])
#     train_logs[row[0]].append(row[1])

# test_logs = dict()
# test = data[data['day'] == 15]
# for row in test[['userID','itemID']].values:
#     test_logs.setdefault(row[0], [])
#     test_logs[row[0]].append(row[1])

# recall(test_logs, rec_dict, train_logs)
# recall(test_logs, {x:[x[0] for x in test_recall_logs[x]] for x in test_recall_logs}, train_logs)