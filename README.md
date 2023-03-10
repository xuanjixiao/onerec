# onerec
在常规推荐系统算法和系统双优化的范式下，一线公司针对单个任务或单个业务的效果挖掘几乎达到极限。从2019年我们开始关注多种信息的萃取融合，提出了OneRec算法，希望通过平台或外部各种各样的信息来进行知识集成，打破数据孤岛，极大扩充推荐的“Extra World Knowledge”。 已实践的算法包括行为数据（多信号，长短期信号），内容描述，社交信息，知识图谱等。在OneRec，每种信息和整体算法的集成是可插拔的，这样的话一方面方便大家在自己的平台数据下灵活组合各种信息，另一方面方便开源共建，大家可以在上边集成自己的各种算法。今天分享的都是之前在线上验证过效果的工作，相关代码和论文已经开源在： https://github.com/xuanjixiao/onerec 。欢迎大家开源共建。

当前代码主要贡献者：[肖玄基](https://scholar.google.com/citations?user=DFsMY2IAAAAJ)([Xuanji Xiao, xuanji.xiao@gmail.com](https://scholar.google.com/citations?user=DFsMY2IAAAAJ))，和子钰(Ziyu He)，戴华强(Huaqiang Dai)，陈华斌(Huabin Chen)、[刘誉臻(Yuzhen Liu)](https://github.com/codestorm04)

# 1）多源信号融合

OneRec系列推荐算法致力于开辟一个新的技术方向--关注多种信息多种信号的萃取/融合，通过所从事业务的视角搜集各种各样的信息来进行信息集成，从而更好的提升系统效果。

算法旨在通过综合利用平台内部的各种信息（包括行为数据，内容描述，社交信息，知识图谱等），创造一个综合的推荐算法。在OneRec，每种信息和整体算法的集成是可插拔的，这样的话一方面方便大家给予自己的平台数据灵活的组合各种信息，另一方面方便开源共建，大家可以在上边集成自己的各种算法。

# 2）合作共建

这个方向理论上是一个很有前景的方向，所以非常欢迎任何人共建，进一步优化OneRec系列算法。


# 3） 目前已经发布的算法：


1） OneRec1_NeighbourEnhancedDNN 行为和内容两种信号的强化建模。增强用户/item的表达和他们的交互.

相关文章：《OneRec2_NeighbourEnhancedyoutubeDNN：基于图的推荐系统Embedding表达增强》，待放出中文链接。  
相关论文： [Neighbor Based Enhancement for the Long-Tail Ranking Problem in Video Rank Models](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_7.pdf)
,Z He, X Xiao, Y Zhou, DLP-KDD 2021

2） OneRec2_Social4Rec 行为/内容之外使用social interest信息。增强用户的表达，有效融合行为，内容，社交兴趣三种信号。  
相关文章：《OneRec3-Social4Rec：社交兴趣增强的推荐算法》 ,待放出中文链接。  
相关论文：https://arxiv.org/pdf/2302.09971.pdf

3） OneRec3_SparseSharing 如何更好的利用点击信号和转化信号。通过彩票理论实现神经元级别的多任务学习，进一步优化cvr的效果。  
相关文章：《OneRec4_LT4REC:基于彩票假设的多任务学习算法》待放出中文链接。  
相关论文：https://arxiv.org/abs/2008.09872


4）OneRec4_SessionLTV 对于一个session浏览过程，结合短期reward和长期reward，来建模用户价值，从而找到LTV价值更高的结果给到用户，在视频场景和google RL simulator上均有正向效果。  
相关论文：https://arxiv.org/pdf/2302.06101.pdf， Oral Presentation at Workshop on Decision Making for Information Retrieval and Recommender Systems in WWW 2023


