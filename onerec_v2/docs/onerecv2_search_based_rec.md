# 方案：

- 当前多数平台都提供了搜索和推荐服务，但是两个服务通常是采用不同的模型分别建模，没有考虑到用户在搜索与推荐两个表示空间中的关联关系，通过构建两者间的联系，能够进一步挖掘用户潜在意图及兴趣，在满足用户需求的基础上，进一步激发用户消费。
  1. 推荐和搜索的本质差异是什么？从用户角度来说，都是用户意图的建模，搜索是单个意图且比较细粒度(少部分意图可能和长期兴趣有区别),推荐是多个意图且是多个层次的。从物料来说，搜索需要处理query,结果也需要query约束。
  2. 可以共享的知识是什么？ 用户多层次的多个意图的意图树，但是搜索可以单分支路由，推荐可以多分支路由。
- 当前构建联系考虑以下方案-：
  1. 构建用户搜索、搜索点击和feed消费混合序列，双向建模序列元素关系和用户意图、偏好表示
  3. 推荐到搜索的转换：
     - 1）信息对齐。可以通过推荐生成一个虚拟的query词汇，从而可以和搜索对齐。对齐方式是，a）虚拟的query词汇和推荐序列，对齐真是的搜索query和序列。2 b）可以有多个query词汇和序列组成的二元组。
  4. 搜索到推荐的转换：
     - 1）兴趣是否一致。query要识别是突发兴趣还是相关兴趣，这样决定query加入到后续兴趣描述的强度。如果query和之前的推荐序列不太相符那就是突发兴趣，和之前的兴趣融合可以坐双兴趣表征；如果相一致，那么可以融合的更加深入一点。总之可以抽离出两个兴趣，突发兴趣和一致兴趣。
     - 2）搜索兴趣随着时间衰减。近期的搜索行为看作成一个trigger，结合当前请求和搜索行为的时间差diff/或s2r后的请求页数，通常，随着diff的增加，该搜索兴趣的强度会逐渐下降。同时，1）对于搜索结果已满足的临时兴趣，没有在后续推荐中继续交互的兴趣，推荐序列中没有对该兴趣进一步加强，随diff增加，也逐渐减弱该兴趣。2）对于搜索结果未完全满足的非短期临时兴趣，在推荐序列中，如产生交互，推荐序列自然会包含该兴趣。
     - 3）多个搜索兴趣的处理。
  5. search,rec, s2r,r2s,这四种结果如何融合？
  6.相似用户行为增强兴趣表征。通过相似用户在s2r或r2s中的行为来增强s2r，r2s表征。从用户搜索序列和feed消费序列中根据序列相似性筛选相似用户，引入相似用户历史序列增强目标用户表示，扩充用户兴趣
  7. 核心问题：找到1)信息提取和融合的创新点 2）和uniSAR的不同创新点。





#### 方案一 @xiangyuan：

- 通过构建S2S、R2R、S2R、R2S四个序列，分别提升搜索和推荐效果，其中R2S和S2R为序列状态转换模块
- 考虑到用户从推荐状态转换到搜索状态通常是由相关推荐引起用户兴趣从而进行搜索，因此在由推荐状态转换到搜索状态时，通过related-select模块对前序推荐序列进行筛选
- 考虑到用户在推荐状态下偏好的物品通常与近期搜索的项目相关，因此在由搜索状态转到推荐状态时，通过设置近邻时间阈值进行过筛

![image](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/img/model1.png)

#### 方案二 @ruixue：

对该问题的思考：因该问题想及时捕捉到用户搜索行为的信号到推荐算法中，因此可将近期的搜索行为看作成一个trigger，结合当前请求和搜索行为的时间差diff/或s2r后的请求页数，通常，随着diff的增加，该搜索兴趣的强度会逐渐下降。同时，1）对于搜索结果已满足的临时兴趣，没有在后续推荐中继续交互的兴趣，推荐序列中没有对该兴趣进一步加强，随diff增加，也逐渐减弱该兴趣。2）对于搜索结果未完全满足的非短期临时兴趣，在推荐序列中，如产生交互，推荐序列自然会包含该兴趣。

- query和item对齐，对比学习loss,但负采样扩大范围，不用batch内。

- 生成两个概率，分别对应偏向搜索兴趣P_s和target_item兴趣偏好分P_t。input：user侧特征，最近的搜索行为时间diff,最近的搜索行为相关item的mean_pooling，与历史行为序列中同类目item的mean_pooling，softmax产出分数。

-  P_s和P_t分别乘对应的最终搜索兴趣表征vs和推荐兴趣表征vr。生成最终兴趣表示emb.

  关于vs和vr的方案

  1）参照UniSAR方式，位置编码改时间差diff。

  2）历史行为序列，两层attention，self-attention后分别用target_item 和 最近搜索行为 做target_atten。

- 对target_item和最近的搜索行为做交叉。

**疑问**：关于最近的搜索行为表示，是否加query_emb，UniSAR在对比loss的sim（a,b）函数用的tanh(aWb^T)方式,query和item的emb可能需要空间转换。但是行为序列中搜索行为用了E_query+Mean(E_i)表示//query_emb+query下点击item mean_pooling.

推荐行为只有E_item，这种方式是否会导致行为序列上，搜索和推荐的表示不一致问题。query和item的emb如果存在空间转换，这样加是否会引入噪声？

#### 公开数据集分析 @zhijian

- 目前数据集: 快手KuaiSAR，美团MT-Small，Amazon（Kindle）
- 美团和亚马逊数据集在item侧特征非常少，找不到推荐 & 搜索序列的共现信息
- 快手KuaiSAR
  - 用户维度做聚合，当前搜索序列 前的推荐序列，时间阈值为1/3/14天的有相同一级类目用户数为26%，40%，60%，

​                 ○ 用户维度做聚合，当前搜索序列 前的推荐序列，时间阈值为1/3/14天的有相同二级类目用户数为24%，35%，52%

​                 ○ 用户维度做聚合，当前推荐序列 前的搜索序列，时间阈值为1/3/14天的有相同一级类目用户数为24%，32%，37%

​                 ○ 用户维度做聚合，当前推荐序列 前的搜索序列，时间阈值为1/3/14天的有相同二级类目用户数为20%，30%，35%

​                 ○ 推荐数据集点击率50%，搜索数据集点击率11%（搜索下的视频会自动播放，所以点击率会偏低）


# related work：

- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- SESRec [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)
- [Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation](https://arxiv.org/pdf/2407.00912)--

# SESRec When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation-2023
- 1）intro：当前做搜推数据结合一起的方法很少，有的也是把数据在一起使用，忽略了两个场景的用户意图不一样。SESRec利用搜索增强推荐场景效果。方法：1）把相似和不相似的用户意图表征解耦 2）把query和item embedding对齐，方便处理query作为用户意图。3）最终得到相似意图，不相似意图，上下文意图三个用户兴趣。. In our paper, we propose a Search-Enhanced framework for the Sequential Recommendation (SESRec) that leverages users’ search interests for recommendation, by disentangling similar and dissimilar representations within S&R behaviors. Specifically, SESRec first
aligns query and item embeddings based on users’ query-item interactions for the computations of their similarities. Two transformer
encoders are used to learn the contextual representations of S&R
behaviors independently. Then a contrastive learning task is designed to supervise the disentanglement of similar and dissimilar
representations from behavior sequences of S&R. Finally, we extract
user interests by the attention mechanism from three perspectives,
i.e., the contextual representations, the two separated behaviors
containing similar and dissimilar interests.
- 2）具体来说，为了解决推荐和搜索行为中的相似兴趣和不相似兴趣，建立搜索序列和推荐序列的相似度矩阵affinity matrix。然后根据这个矩阵的得分，对搜索（推荐）序列提取出来和对方相似的序列p和不相似的序列n，这样搜索（推荐序列）可以分解为3个序列，原始序列，相似序列，不相似序列。 对于6各序列使用target attention。对于搜索序列的处理，使用query和item对比学习，映射到同一个空间。
<img width="500" alt="image" src="https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/img/IMG_8218.jpeg">


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



