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
            │      3_generate_original_matrix.ipynb      # 快速相似度矩阵运算方法
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


## Q&A

> **Q:** 我是香港中文大学（深圳）数据科学专业的学生，想报名参加这个比赛作为毕业项目的，但是超过了比赛的报名时间，所以想问一下你能不能发我一下比赛的原始数据呢？          
> **A:** 数据集较大无法存储到Github，请在网盘链接: https://pan.baidu.com/s/1Mnp4R27qXt_b367G4EcVaA 提取码: 5ecq下载

> **Q:** 看你们的复赛方法介绍，讲到了也试过word2vec学习embedding，然后用faiss来做召回。请问这样的方法的效果和你们最后用的item CF的方法，比较起来如何呢？         
> **A:** 很抱歉，由于时间关系我们没有进行对比，因为itemCF已经取得了比较好的效果。我们尝试过embedding+faiss的方案是可行的。为了后期快速搭建线上pipeline我们选择了更为简单的itemCF。但在2020KDDCUP中，我们分别尝试了ItemCF与word2vec+faiss方案，ItemCF取得了更好的召回效果，但embedding可以作为很好的特征。请查阅我库中2020KDDCUP项目

> **Q:** 按照方案中描述：使用了用户活跃度的置信度计算 Item CF，这里的sim(i,j)!=sim(j,i)，但是代码中这样看应该是相等的？另外，改进的相似度方法中，公式和代码对不上。是需要进一步推导嘛？       
> **A:** 谢谢你的邮件！此步类似于统计共现次数，统计的mat\[a, b]并不是最终的相似度，在后续得到a到b的相似度时使用的是mat\[a, b] / f(a)的计算方法。请您再看一下代码和相应的公式。如果还有更多的疑问可以再联系我。

> **Q:** 在对行为做临近时间加权的时候，好像这样更好， data['behavior'] = data['behavior'] / (max_day-data['day'])，这个不知道你们有没有调整过。        
> **A:** 我们的时间权重设置是按照11年kdd cup第一名的方式设置的，同时也是SVDFeature中temporal SVD的设置方式。我们没有进行你们的尝试，如果data['behavior'] = data['behavior'] / (max_day-data['day'])效果更好，可能说明在不同数据集上要多尝试几种不同的设置方法，然后选择最好的方法。

> **Q:** 最后一个问题是关于线下的novel recall@50的计算，我们用复赛的训练数据除去最后一天也就是第15天的数据做训练，然后用初赛round b的测试数据的第15天做验证，得到的novel recall @50大概是0.039，这个和你们文档中说的只用召回代码就可以得到0.053的结果有点差距。请问你们还又做了些什么处理呢？如果不做排序的话。当然也有可能是我们线下计算的指标和比赛的时候线上算的指标不太一致，不知道你们有没有碰到过类似的问题。      
> **A:** 线上效果0.053对应在testB上的验证效果是0.0385，可能是由于线上线下数据集的大小不同或者分布不同导致，但可以保证的是通过testB验证的结果与线上结果是同步增长。在开源代码中的评价方式与线上的指标是相同的。









## 声明
本项目库专门存放CIKM2019挑战赛的相关代码文件，所有代码仅供各位同学学习参考使用。如有任何对代码的问题请邮箱联系：cs_xcy@126.com

If you have any issue please feel free to contact me at cs_xcy@126.com

天池ID：BruceQD & 人畜无害小白兔 & **小雨姑娘**
