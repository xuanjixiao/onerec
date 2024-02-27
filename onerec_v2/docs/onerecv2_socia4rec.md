owner by DR Wanglin

1 设计概述
社交和图的建模方案确定。socialLGN + contrative learning.  
overleaf 文档链接 https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4 正在调试代码，下一步需要观察 social 的对比学习信息对推荐性能的影响到底有多大  —wanglin

n social recommendation systems, the heterogeneity of graph data introduces noise prop-
agation when graph convolution is used for information propagation. For example, users
who are friends both in social networks and in the real world tend to have higher similarity,
and their information propagation is considered beneficial for improving the performance
of the recommendation system. However, the social relationships between celebrities and
their fans in social networks cannot be simply categorized as friendships, and the informa-
tion propagation between them needs to be carefully considered to avoid degrading the
performance of the recommendation system.
On the other hand, considering the sparsity of social data, there is a significant propor-
tion of users who have limited social connections and interactions with items. To enhance
the representation capability of this subset of data, we adopt a multi-view approach for
data augmentation, which involves first applying low-rank SVD to the user-item matrix
to obtain user embeddings. The advantage of low-rank decomposition is the ability to
select principal components and remove unnecessary interference, effectively filtering the
user-item matrix. Then, the user embeddings are used for dot product operations to calcu-
late the similarity between all pairs of users. By setting a threshold, a reconstructed social
graph is generated. In addition to the original social graph, there is a new reconstructed
view of the social graph. The contrastive learning between these two views can be seen as
data augmentation for users, alleviating the challenges imposed by graph sparsity.
2 Preliminaries
2.1 Social recommendation
2.2 Contrastive learning
References
1
‘![image](https://github.com/xuanjixiao/onerec/assets/15994016/c1aa76bd-b464-46ab-bd15-2523bfecd7af)


2 试验进展



