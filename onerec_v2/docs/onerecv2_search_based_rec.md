# 方案：

- 当前多数平台都提供了搜索和推荐服务，但是两个服务通常是采用不同的模型分别建模，没有考虑到用户在搜索与推荐两个表示空间中的关联关系，通过构建两者间的联系，能够进一步挖掘用户潜在意图及兴趣，在满足用户需求的基础上，进一步激发用户消费。当前构建联系考虑以下方案：
  1. 构建用户搜索、搜索点击和feed消费混合序列，双向建模序列元素关系和用户意图、偏好表示
  2. 从用户搜索序列和feed消费序列中根据序列相似性筛选相似用户，引入相似用户历史序列增强目标用户表示，拓展用户兴趣


# related work：

- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)
- [Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation](https://arxiv.org/pdf/2407.00912)--
-
# 2024 UDITSR-对于推荐的每一次交互生成虚拟的query
对于推荐的每一次交互生成虚拟的query，然后再搜推场景上对于user-query-item这样的三元图，进行类似transE一样的训练。美团，理论完备，实验对比不充分（只有2个属于同领域工作，但是这两个还比较古老），效果提升只有1%。

 <img width="500" alt="image" src="https://github.com/user-attachments/assets/857114ab-628c-433e-9b2e-ceac93228b41">

  
  1) 动机：用户有两种兴趣，推荐常见的隐式的不变的固有兴趣和搜索常见的变化的显式的需求兴趣（unchanging inherent intents and changing demand
intents）。其实这两种兴趣在两个场景中都存在。比如tom和amy固有兴趣分别喜欢辣和甜，但是变化是虽然都来到了pizza hub但是amy今天想要吃pizza而tom想吃pasta。或者说喜欢便宜，但是夏天喜欢tshirt，冬天喜欢雪地靴。对于推荐来说，需要用搜索的demand来监督兴趣学习；对于搜索来说，需要用固有兴趣来个性化结果。这两块当前都做的不够。r, both types of intents are implicitly expressed in recommendation scenario, posing challenges
in leveraging them for accurate intent-aware recommendations.users express their demand
intents explicitly through their query words。
- 挑战：: (1) accurately modeling users’ implicit demand intents in recommendation; (2) modeling the relation between the dual intents and the interactive items。使用搜索query监督推荐中的主动兴趣，建模<inherent intent, demand intent, interactive item>三元组。效果在美团上GMV和点击率+1%。To accurately simulate users’ demand intents in recommendation, we utilize real queries
from search data as supervision information to guide its generation. To explicitly model the relation among the triplet <inherent intent, demand intent, interactive item>, we propose a dual-intent translation propagation mechanism to learn the triplet in the same semantic space via embedding translations。
- 方法：
1) 构图U——I，在推荐域中生成虚拟的query作为边的属性，信息来源使用user embedding，item embedding，user带有的query信息，item带有的query信息，eˆ𝑞 = MLP(e𝑢 ∥e𝑖 ∥e𝑞𝑢∥e𝑞𝑖),。这个query就是demanding intent的表示。胜 How to accurately model a user’s implicit demand intent in recommendation with search data? 使用搜索query监督生成推荐中的demand interest，之前的工作假设demand interest是不变的，这个是不对的，应该使用一个变化的query 历史序列。
<img width="146" alt="image" src="https://github.com/user-attachments/assets/d5a2f1ad-7929-4ea6-86f0-cc6afd2b24cb">
<img width="161" alt="image" src="https://github.com/user-attachments/assets/860d14e6-e585-4830-9fce-502cf140c059">
2) 建模dual-intent和item关系得到user和item embeding，在user-query-item这样的带有边属性的三元图上使用类似gcn+transE算法。 How to couple the dual intents to model the relation among the intents and the interactive items? 如何建模两个意图和item的关系。
<img width="208" alt="image" src="https://github.com/user-attachments/assets/400ab845-23a2-4345-b2c1-161a27808c4a">
<img width="285" alt="image" src="https://github.com/user-attachments/assets/0919a23d-95b7-49e7-a5d2-c4b30e4c39d8">

3）最后使用user，item，query做预测。
<img width="95" alt="image" src="https://github.com/user-attachments/assets/4e0cb878-aafc-4d98-89c0-93319be260d4">


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



