# onerec
Under the paradigm of conventional recommendation system algorithm and system dual optimization, the effect mining of a single task or a single business by front-line companies has almost reached its limit. Since 2019, we have started to pay attention to the extraction and fusion of various information, and proposed the OneRec algorithm, hoping to integrate knowledge through platforms or various external information, break data silos, and greatly expand the recommended "Extra World Knowledge". Algorithms that have been practiced include behavioral data (multi-signal, long-term and short-term signals), content description, social information, knowledge graph, etc. In OneRec, the integration of each type of information and the overall algorithm is pluggable. In this way, on the one hand, it is convenient for everyone to flexibly combine various information under their own platform data, and on the other hand, it is convenient for open source co-construction. You can integrate your own on it. various algorithms. What I share today is the work that has been verified online before. The relevant codes and papers have been open sourced at: https://github.com/xuanjixiao/onerec. Welcome everyone to contact us, join the WeChat group (contact information: growj@126.com), and join the team for open source co-construction.

在常规推荐系统算法和系统双优化的范式下，一线公司针对单个任务或单个业务的效果挖掘几乎达到极限。从2019年我们开始关注多种信息的萃取融合，提出了OneRec算法，希望通过平台或外部各种各样的信息来进行知识集成，打破数据孤岛，极大扩充推荐的“Extra World Knowledge”。 已实践的算法包括行为数据（多信号，长短期信号），内容描述，社交信息，知识图谱等。在OneRec，每种信息和整体算法的集成是可插拔的，这样的话一方面方便大家在自己的平台数据下灵活组合各种信息，另一方面方便开源共建，大家可以在上边集成自己的各种算法。今天分享的都是之前在线上验证过效果的工作，相关代码和论文已经开源在： https://github.com/xuanjixiao/onerec 。欢迎大家联系我们，加入微信群（联系方式：growj@126.com），加入团队进行开源共建。

The main contributor to the current code: [Xuanji Xiao, xuanji.xiao@gmail.com](https://scholar.google.com/ citations?user=DFsMY2IAAAAJ)), Ziyu He, Huaqiang Dai, Huabin Chen, [Yuzhen Liu](https://github.com/codestorm04)
当前代码主要贡献者：[肖玄基](https://scholar.google.com/citations?user=DFsMY2IAAAAJ)([Xuanji Xiao, xuanji.xiao@gmail.com](https://scholar.google.com/citations?user=DFsMY2IAAAAJ))，和子钰(Ziyu He)，戴华强(Huaqiang Dai)，陈华斌(Huabin Chen)、[刘誉臻(Yuzhen Liu)](https://github.com/codestorm04)

# 1）多源信号融合

OneRec系列推荐算法致力于开辟一个新的技术方向--关注多种信息多种信号的萃取/融合，通过所从事业务的视角搜集各种各样的信息来进行信息集成，从而更好的提升系统效果。

算法旨在通过综合利用平台内部的各种信息（包括行为数据，内容描述，社交信息，知识图谱等），创造一个综合的推荐算法。在OneRec，每种信息和整体算法的集成是可插拔的，这样的话一方面方便大家给予自己的平台数据灵活的组合各种信息，另一方面方便开源共建，大家可以在上边集成自己的各种算法。

# 2）合作共建

这个方向理论上是一个很有前景的方向，所以非常欢迎任何人共建，进一步优化OneRec系列算法。


# 3） 目前已经发布的算法：


1） OneRec1_NeighbourEnhancedDNN 行为和内容两种信号的强化建模。增强用户/item的表达和他们的交互.

相关文章：《OneRec1_NeighbourEnhancedyoutubeDNN：基于图的推荐系统Embedding表达增强》，待放出中文链接。  
相关论文： [Neighbor Based Enhancement for the Long-Tail Ranking Problem in Video Rank Models](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_7.pdf)
,Z He, X Xiao, Y Zhou, DLP-KDD 2021

2） OneRec2_Social4Rec 行为/内容之外使用social interest信息。增强用户的表达，有效融合行为，内容，社交兴趣三种信号。  
相关文章：《OneRec2_Social4Rec：社交兴趣增强的推荐算法》，待放出中文链接。   
相关论文：https://arxiv.org/pdf/2302.09971.pdf WISE2023

3） OneRec3_SparseSharing 如何更好的利用点击信号和转化信号。通过彩票理论实现神经元级别的多任务学习，进一步优化cvr的效果。  
2018 CVR多任务工作：[Calibration4CVR：2018年关于“神经元级别共享的多任务CVR”的初探-2018](https://zhuanlan.zhihu.com/p/611453829)  

2020 [《OneRec3_LT4REC:基于彩票假设的多任务学习算法》](https://mp.weixin.qq.com/s/4PO6EK3b4VCKO0ibd76C9w)。  相关论文：https://arxiv.org/abs/2008.09872   ECIR2023

2023CVR多任务工作：[Click-aware Structure Transfer with Sample Weight Assignment for Post-Click Conversion Rate Estimation](https://arxiv.org/abs/2304.01169), ECML-pkdd 2023.  
2023 用户生命周期视角下的多任务推荐模型 [STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation](https://arxiv.org/abs/2306.12232) Recsys2023



4）OneRec4_SessionLTV 对于一个session浏览过程，结合短期reward和长期reward，来建模用户价值，从而找到LTV价值更高的结果给到用户，在视频场景和google RL simulator上均有正向效果。  
相关论文：https://arxiv.org/pdf/2302.06101.pdf， WWW 2023


# 4）OneRec V2
最近准备正式开始 OneRec 2期项目迭代，主要可能集中在社交信息融合，搜索和推荐融合，多模态融合，跨业务夸场景融合这4个方向。大家可以看看有没有感兴趣的方向可以参与，或者跟我聊。

产出主要是论文和线上效果。

下周我们正式开个短会对齐下，会之前大家可以畅所欲言，或者跟我聊想法都可以

整体节奏会比较慢，可能一个月对齐一次进度

onerec一期：https://github.com/xuanjixiao/onerec
