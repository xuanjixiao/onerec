# 方案：

- 当前多数平台都提供了搜索和推荐服务，但是两个服务通常是采用不同的模型分别建模，没有考虑到用户在搜索与推荐两个表示空间中的关联关系，通过构建两者间的联系，能够进一步挖掘用户潜在意图及兴趣，在满足用户需求的基础上，进一步激发用户消费。当前构建联系考虑以下方案：
  1. 构建用户搜索、搜索点击和feed消费混合序列，双向建模序列元素关系和用户意图、偏好表示
  2. 从用户搜索序列和feed消费序列中根据序列相似性筛选相似用户，引入相似用户历史序列增强目标用户表示，拓展用户兴趣


# related work：

- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)
- [Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation](https://arxiv.org/pdf/2407.00912)
  1) 动机：推荐的固有兴趣和搜索的主动兴趣。这个兴趣会同时影响用户的交互行为。r, both types of intents are implicitly expressed in recommendation scenario, posing challenges
in leveraging them for accurate intent-aware recommendations.users express their demand
intents explicitly through their query words。
挑战：: (1) accurately modeling users’ implicit demand intents in recommendation; (2) modeling the relation between the dual intents and the
interactive items
2）方法：使用搜索query监督推荐中的主动兴趣，建模<inherent intent, demand intent, interactive item>三元组。效果在美团上GMV和点击率+1%。To accurately simulate users’ demand intents in recommendation, we utilize real queries
from search data as supervision information to guide its generation. To explicitly model the relation among the triplet <inherent intent, demand intent, interactive item>, we propose a dual-intent translation propagation mechanism to learn the triplet in the same semantic space via embedding translations。
 

-  Joint Search and Recommendation. In recent years, there hasbeen a trend toward integrating S&R. These works primarily fall
into two categories: (a) Search enhanced recommendation [14, 25,
27, 30, 36]. This type of work utilizes search data as supplementary information to enhance the recommendation performance.
IV4Rec [25, 26] utilizes causal learning, treating user-searched
queries as instrumental variables to reconstruct user and item
embeddings. Query-SeqRec [14] is a query-aware model which
incorporates user queries to model users’ intent. SESRec [27] uses
contrastive learning to disentangle similar and dissimilar interests
between user S&R behaviors. (b) Unified S&R [11, 38, 40, 41, 43].
This kind of work performs joint learning of S&R to enhance the
model performance in both scenarios. JSR [41, 42] simultaneously
trains two models for S&R using a joint loss function. USER [40]
integrates user S&R behaviors and feeds them into a transformer
encoder. UnifiedSSR [38] proposes a dual-branch network to encode
the product history and query history in parallel. In this paper, we
develop a framework that

# UniSAR
主要是建模s2s,r2r,s2r,r2s四个序列，促进搜索和任务效果都提升。
方法：
1）整体上使用extract, alignment, fusion三个极端。
2）使用attention构建这四个序列表征，其中s2r和r2s注意只提取不相同的行为作为attention计算（即搜索和推荐相间的两个行为），技巧使用multihead self attention和mask掉相同场景（都是搜索或者推荐））的行为；
3）s2r和r2r构建产生推荐表征Vr，r2s和s2s产生搜索表征Vs。技巧使用对比学习使得s2r和r2r相似，r2s和s2s相似;cross attention进行两者信息融合。
4）其他：为了对齐query和item，对query和item进行对比学习。
![image](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/img/IMG_6874.jpeg)

使用KuaiSAR真实数据集合和AMAZON合成数据集合，在推荐和搜索上的hitrate 和 ndcg上有显著提升，其中推荐的提升更大。



