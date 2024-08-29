owner by DR Wanglin

## 1 设计概述
社交和图的建模方案确定。socialLGN + contrative learning.  
1)当前社交方式中，相似朋友会有相似兴趣，但是单向关注的大v关系不能说有相似兴趣。其次，存在行为或者社交关系很稀疏的情况。方法是建立一个新的社交关系图。先通过svd提取user-item的embedding，这样可以提取重要信息且减少噪音。然后使用user embedding相似度建立新的图，基于行为的关系图和原来的社交关系图的对比学习，可以当作用户的数据增强。
如下图所示，社交关系是一个图，然后user-item是一个图，然后通过user-item提取得到的user是一个user图。loss除了user-item之外，还有一个就是social graph得到的user embedding和行为得到的user embedding进行contrastive learnin

2）contribution够吗？  
 a)如何处理social node with few social connections？ diffusion model
 b) 另外，没有使用user-user-item这样的信息，第一个链接是社交关系，第二个链接是行为关系，这个信息应该能更好的捕捉到朋友的商品兴趣。可以增加一个view，之前是3个view，社交的user-user，行为的user-user和user-item，现在增加一个社交的user-item。

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

新方案？


## 2 试验进展 (0402)

论文原文中LastFM的实验数据和加入对比损失之后的实验效果(对比损失系数 1e-7)

| Dataset | Metric | SocialLGN |SocialLGN-CL
| -------- | -------- | -------- |-----------
| LastFM     | Precision@10     | **0.1972**     | 0.1968
| -     | Precision@20     | 0.1368     |**0.1375**
| -     | Recall@10     | **0.2026**     |0.2017
| -     | Recall@20     | 0.2794     |**0.2805**
| -     | NDCG@10     | **0.2566**     |0.2557
| -     | NDCG@10     | **0.2883**     |0.2822

目前的实验结果显示 SocialGraph 对比损失的加入对结果影像很小，只有在对比损失占比较小的时候会表现出较好的性能，考虑可以改进的方向有以下几个方面：
* 1）稠密social graph&稀疏行为数据 可能才会起作用
* 2）t-SNE 可视化user embedding
* 3）用户分层
* 4）@50
* 5）换数据集
* 6）social graph去噪声

## 3 下一步需要做的事情：
  * 1）解决 SVD oom 的问题（并行，或者只对 social graph 进行 SVD 分解）
  * 2）找社交信息包含较多噪声的数据集进行实验

其他：比较漂亮的图：
<img width="280" alt="image" src="https://github.com/user-attachments/assets/6cbe77d1-29a3-4ec0-875d-7d9730b255c0">

