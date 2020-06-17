# CIKM-2019-AnalytiCup
2019-CIKM挑战赛，超大规模推荐之用户兴趣高效检索赛道 冠军解决方案

This repository contains the champion solution on CIKM 2019 EComm AI - Efficient and Novel Item Retrieval for Large-scale Online Shopping Recommendation Challenge.

## 解决方案blog

知乎文章：https://zhuanlan.zhihu.com/p/91506866

## 文件结构

    │  LICENSE
    │  project_structure.txt
    │  README.md
    │  初赛方案简介.pdf
    │  复赛方案简介.pdf
    │  答辩ppt.pptx
    │  
    ├─Qualification                                        # 初赛解决方案
    │      Qualification.py
    │      
    └─Semi-Finals                                          # 复赛解决方案
        ├─online_recommendation                            # 生成线上结果
        │      dockerfile
        │      downward_map.zip
        │      lgb_0924_1652
        │      model0924_base.file
        │      read_me.txt
        │      run.sh
        │      test.py
        │      upward_map.zip
        │      
        └─underline_trainning                              # 生成线下验证结果以及特征
            │  Readme.pdf
            │  
            ├─Step1 itemCF_based_on_Apriori                # 基于Apriori关联规则法生成商品关联矩阵
            │      1_generate_user_logs.ipynb
            │      2_generate_hot_table.ipynb
            │      3_generate_original_matrix_0.ipynb      # 采用并行方法加速商品矩阵运算（ipynb不支持multiprocessing并行，py文件可以）
            │      3_generate_original_matrix_1.ipynb
            │      3_generate_original_matrix_2.ipynb
            │      3_generate_original_matrix_3.ipynb
            │      3_generate_original_matrix_4.ipynb
            │      3_generate_original_matrix_5.ipynb
            │      3_generate_original_matrix_6.ipynb
            │      3_generate_original_matrix_7.ipynb
            │      3_generate_original_matrix_8.ipynb
            │      4_Merge.ipynb
            │      5_Save_sparse_to_dense.ipynb
            │      6_Sta_for_SparseMatrix.ipynb            # 将稀疏的关联矩阵转化为Hash结构以加快检索效率
            │      7_generate_recall.ipynb                 # 基于关联矩阵为每个用户生成candidate列表
            │      
            ├─Step2 Generate_feature_for_Ranking           # 为candidate列表生成特征
            │      1_generate_static_features.ipynb
            │      2_generate_dynamic_feature.ipynb
            │      3_generate_time_feature.ipynb
            │      
            └─Step3 Ranking                                # 基于candidate列表与特征做出推荐
                    1_build_model.ipynb
                    2_recommendation.ipynb
                    
    注意！有些文件较大未上传到github，除数据集外，所有缺失文件均可在代码中生成。

## 声明
本项目库专门存放CIKM2019挑战赛的相关代码文件，所有代码仅供各位同学学习参考使用。如有任何对代码的问题请邮箱联系：cs_xcy@126.com

If you have any issue please feel free to contact me at cs_xcy@126.com

天池ID：BruceQD & 人畜无害小白兔 & **小雨姑娘**
