# 方案：
* MMRec Insights初步idea
协同过滤成为了学习用户和项目潜在表示(lantent representatinos)的一个重要方式。 其中使用GCN并结合BPR损失是一种常见的做法。但是交互数据的稀疏性会导致这个方法有两个缺点：1）我们只能观察到一部分交互行为，大部分交互行为并没有被观察到，GCN通常的做法是在没有发现交互行为的数据中随机负采样，这样就会容易将没被观察到的交互行为误认为是负样本。2）只能通过少量的交互行为学习用户和项目的潜在表示，无法发掘item或用户内在的特征。为了解决这两个问题，打算从两个角度出发，首先使用只用正样本的自监督学习，通过只构建正样本对进行对比学习[借鉴CV及nlp中的方法]；对于第二个问题，使用样本的多模态特征，提升item的潜在表示。

对于自监督学习，已经有研究将CV领域中的BYOL和simsiam及nlp中的simcse方式应用于推荐问题，我会尝试将DINO中的想法（self-distill，使用前一个epoch的student网络作为teacher，计算高效+效果更好）和VICReg中的loss【covariance term; invariance term】
* 对于图模型结构：尝试llayergcn
* 对于多模态特征：设计一个item2item网络，因为LATTICE中提出的item2item网络没有效果，反而会降低效果，FREEDOM中有做实验，但是这篇论文并没有解决这个问题，只是冻结了这个网络。 
* 设计多模态和id特征融合网络。这个网络对最终结果也比较重要【MMGCN中有提到】。


1）不同的跳与跳相似。可以作为自监督的信号
2）user2item 与item2item网络的交互。





目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中  
方案：
1. 由于前任设计的多模态item2item图模型不太理想，故尝试设计一个高效的item2item的图学习方式。初步的方案可以使用行为数据学习的图模型作为辅助，交叉融合更新行为>学习图模型和多模态图模型。
2. 结合CV领域中自监督的方式，并结合图模型的特性，设计一个只用正样本学习的自监督学习方式，提高模型学习的效率
3. 探索行为图模型和多模态图模型的融合方式，提高最终的效果
4. 相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2
数据集合：
Amazon  baby数据集合
方案：探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2。


# 参考文档
* MMGCN首次使visual，text，acoustic三种模态特征在行为图上进行学习。
* GRCN去掉了有噪声的行为边。
* GMAT用了attetion处理了不同模态的重要性
* DMTL使用解耦的representation和attention处理不同模态关系。
* MGCL 使用了行为信息来提纯内容信息。用了对比学习来进行数据增强，同时保证每个模态的表达.使用了
