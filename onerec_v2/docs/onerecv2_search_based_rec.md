# 方案：

- 当前多数平台都提供了搜索和推荐服务，但是两个服务通常是采用不同的模型分别建模，没有考虑到用户在搜索与推荐两个表示空间中的关联关系，通过构建两者间的联系，能够进一步挖掘用户潜在意图及兴趣，在满足用户需求的基础上，进一步激发用户消费。当前构建联系考虑以下方案：
  1. 构建用户搜索、搜索点击和feed消费混合序列，双向建模序列元素关系和用户意图、偏好表示
  2. 从用户搜索序列和feed消费序列中根据序列相似性筛选相似用户，引入相似用户历史序列增强目标用户表示，拓展用户兴趣


# related work：

- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)
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

![image](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/img/IMG_6874.jpeg)


