# 1 meeting 202307--talking about the future directions and references

1.1 社交&行为. social4rec v2--Fuse social network information and behaviors signal. -huaqiang/andy/wanglin
1）主要方案：	
	A)排序--升级v1版本算法。社交信息单簇（一个人可以从属多个兴趣组）变多簇，多簇（社交域）对多峰（行为域）。 —华强。
	B)召回----社交信息多簇和图联通方式进行召回 --andy
	C)其他。社交信息以一种指导或者修正的方式加进去，比如类似NLP大模型推理的时候可能需要加一句话作为指导（有些算术题目，如果加上let’s step by step 就会得到更加正确的答案）——和欧阳合作。参考：Anoop Deoras (Amazon)--Topic: Zero Shot Recommenders, Large Language Models and Prompt Engineering 降了不少跨域的， 
	  采用自监督对比学习对社交数据进行增强，具体而言：1）采用 SVD 分解对 U-I 矩阵进行分解，得到 user 的 embedding，  2）利用 user embedding 进行点积运算，设置相似度阈值，重构一个新的 U-U social graph，3）将重构的social graph 与已知的 social graph 进行对比学习，以增强推荐性能。
2）问题: a) the noise of info passing in graph? we can use external info/knoledge or node's info as supervisory signals to calibrate the graph connection confidence. Or we can use self-supervised learning. 
	b) All in all, we want to learning a better info extraction of the graph and then fuse the knowledge with behaviors. Refer: dual-learning,Ripple-net,social4rec v1.
	c) SVD is a computational-intensive algorithom, which is not suitable for the industrial senarios. We need to find the substitute or reduce the computational complexity.

3）参考References: 
	a) 基于社交兴趣增强的视频推荐算法 https://zhuanlan.zhihu.com/p/639351979
	b) 公开数据集合-tenrec: 1)the dataset contains social and behavior data 2)intro: https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
	c) LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation
	d) Socially-Aware Self-Supervised Tri-Training for Recommendation

1.2 Search based rec—qichao。 确定性行为和随机行为，nlp和行为。搜索推荐做mmoe，两个expert。

1.3 多模态&行为-wenqi。
1）方案：
2）问题：1)相同图像的ctr(点击行为）分布差异比较大，融合进行为为主模型问题。比如白牌服装和品牌服装长的一样。行为-买锅推勺子，mmu-买锅推锅 2）问题，低级表征和高级表征。低级表征，embedding经常拿不到效果，hulu经历把embedding变成标签才拿到效果。

1.4 multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。
问题：1）比如直播优势类目是衣服，电商优势类目是食品。分布差别大，数据聚合起来知识比较多。2）不同域的序列信息加进来，兴趣不一定一致，比如商品域经常出现偏热门问题（淘宝直播使用其他域信息发现，商品域pvr很高偏热门）。--wenhao/bangzheng 

# 2 meeting note-20230905.
2.1 社交&行为：
1）下次会议：user-user关系置信度，然后修正原始的结果。dongqian说的置信度文章。-Andy
2）下次会议：图关系提取，和 融合进入其他模型比如推荐模型—wanglin
3）下次会议：调研社交用在排序具体方案？—yuqiao/华强

2.2 搜索信号运用到推荐里面--qichao
1. 特征: 搜索信号行为序列特征直接运用到模型里面
2. 特征: 搜索信号具体行为序列与推荐里面曝光未点击行为序列通过对比学习建模负向行为
3. 特征: 搜索信号可以作为query去检索长序列, 来建模长序列建模
4. 搜索场景样本训练模型, 产出emb 给推荐场景使用
5. 搜索场景与推荐场景联合建模, 会比较麻烦
下次会议：确定具体方案

2.3 多模态&行为-wenqi
下次会议：需要出具体方案，分享，

2.4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。
下次会议：具体方案确定，包括多个域的序列如何融合--wenhao


# 3 meeting 20231010
1） 社交&行为：
社交和图的建模方案确定。链接？下次会议：初步开发代码在公开数据集合尝试。—wanglin，
下次会议：调研社交用在排序具体方案。—yuqiao/华强

2.2 搜索信号运用到推荐里面--qichao
1. 方案: 搜索信号具体行为序列与推荐里面曝光未点击行为序列通过对比学习建模负向行为。链接：下次：方案需要细化。

2.3 多模态&行为-wenqi
目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中
方案：1）由于前任设计的多模态item2item图模型不太理想，故尝试设计一个高效的item2item的图学习方式。初步的方案可以使用行为数据学习的图模型作为辅助，交叉融合更新行为>学习图模型和多模态图模型。
      2）结合CV领域中自监督的方式，并结合图模型的特性，设计一个只用正样本学习的自监督学习方式，提高模型学习的效率
      3) 探索行为图模型和多模态图模型的融合方式，提高最终的效果
      4)相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2

2.4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。
方案：使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。
下周：和zhuoxi一起细化方案.


# 4 meeting 20231031
1） 社交&行为：
社交和图的建模方案确定。链接？下次会议：初步开发代码在公开数据集合尝试。—wanglin，
下次会议：调研社交用在排序具体方案。—yuqiao/华强。 用行为信息预测社交信息，用社交信息预测行为信息，最后融合。调研

2.2 搜索信号运用到推荐里面--qichao
1. 方案: 搜索信号具体行为序列与推荐里面曝光未点击行为序列通过对比学习建模负向行为。链接：下次：方案需要细化。
确实是有点对比学习的意思：使用 target-item 从 pos_seq 提取出emb是跟 target-item相关的pos_seq-item，使用 target-item 从 neg_seq 提取出emb是跟 target-item相关的 neg_seq-item，之后计算两者：pos_seq-item和neg_seq-item 互信息更大
pos_seq-item 和 neg_seq-item 提取就相当于对比学习里面的信息增强。
让序列特征emb学习的更好

2.3 多模态&行为-wenqi
目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中
方案：1）由于前任设计的多模态item2item图模型不太理想，故尝试设计一个高效的item2item的图学习方式。初步的方案可以使用行为数据学习的图模型作为辅助，交叉融合更新行为>学习图模型和多模态图模型。
      2）结合CV领域中自监督的方式，并结合图模型的特性，设计一个只用正样本学习的自监督学习方式，提高模型学习的效率
      3) 探索行为图模型和多模态图模型的融合方式，提高最终的效果
      4)相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2

2.4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi/kexin/wenhao
方案：使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。
下周：和zhuoxi一起细化方案.


# 5 meeting 20231128
1） 社交&行为：
社交和图的建模方案确定。socialLGN + contrative learning.  overleaf 文档链接 https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4  —wanglin
下次会议：调研社交用在排序具体方案。—yuqiao/华强。 用行为信息预测社交信息，用社交信息预测行为信息，最后融合。调研

2.2 搜索信号运用到推荐里面--qichao
1. 方案: 搜索信号具体行为序列与推荐里面曝光未点击行为序列通过对比学习建模负向行为。
细化的方案：？

2.3 多模态&行为-wenqi
目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中。
方案：探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2

2.4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi/kexin/wenhao
方案：使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。
下周：和zhuoxi一起细化方案. recbole。 补一张图？


# 6 meeting 20230109
1） 社交&行为：
社交和图的建模方案确定。socialLGN + contrative learning.  overleaf 文档链接 https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4 正在调试代码，下一步需要观察 social 的对比学习信息对推荐性能的影响到底有多大  —wanglin
下次会议：调研社交用在排序具体方案。—guanxin/华强。 用行为信息预测社交信息，用社交信息预测行为信息，最后融合。调研

2.2 搜索信号运用到推荐里面--qichao
1. 方案: 搜索信号具体行为序列与推荐里面曝光未点击行为序列通过对比学习建模负向行为。
细化的方案：1）补充方案 2）参考链接 3）结果。

2.3 多模态&行为-wenqi
目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中。
方案：探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2。

2.4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi/kexin/wenhao
方案：使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。
下周：和zhuoxi一起细化方案. recbole。 补一张图？

# 7 meeting 20240227
## 1 社交&行为：
* 本次进展：在 LastFM 数据集上的结果表现不佳，加入对比损失反而会降低推荐的性能，下一步需要从以下几个方面分析新加入的对比损失对社交推荐的作用: 1) 稀疏 social graph 可能才会起作用; 2) t-SNE 可视化user embedding; 3) 用户分层; 4) @50; 5) 换数据集.
* 下次预期：调研社交用在排序具体方案。—guanxin/华强。 用行为信息预测社交信息，用社交信息预测行为信息，最后融合。调研
* 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
* 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。
  
## 2 搜索信号运用到推荐里面--qichao
* 本次: 
* 下次：1）补充方案 2）参考链接 3）结果。
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md)
* 方案简介：搜索信号具体行为序列（正信号），与推荐里面曝光未点击行为序列（负信号）通过对比学习建模负向行为。

## 3 多模态&行为-wenqi 
* 本次：将item的多模态特征/语义特征融入到user2item的行为数据学习当中。
* 下次：探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md) , [tencent doc](https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2)。
* 方案简介：GCN做推荐问题有两个：1）随机负采样，其实是不准确的。2）正样本很稀疏。解决办法：1）用正样本的自监督学习DINO（不同的跳与跳相似。可以作为自监督的信号），通过只构建正样本对进行对比学习[借鉴CV及nlp中的方法；2）使用样本的多模态特征，提升item的潜在表示-FREEDOM i2i。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
* 本次：和zhuoxi一起细化方案. recbole。 补一张图？
* 下次：
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
* 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用


# 7 meeting 20240402
## 1 社交&行为：
* 本次进展：在 LastFM 数据集上的结果表现不佳，加入对比损失反而会降低推荐的性能，下一步需要从以下几个方面分析新加入的对比损失对社交推荐的作用: 1) 稀疏 social graph 可能才会起作用; 2) t-SNE 可视化user embedding; 3) 用户分层; 4) @50; 5) 换数据集. 
* 下次预期：1）调研社交用在排序具体方案。—guanxin/华强。 用行为信息预测社交信息，用社交信息预测行为信息，最后融合。调研。 2）wanglin：@svd分解oom'问题
* 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
* 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。
  
## 2 搜索信号运用到推荐里面--qichao
* 本次: 
* 下次：1）补充方案 2）参考链接 3）结果。
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md)
* 方案简介：搜索信号具体行为序列（正信号），与推荐里面曝光未点击行为序列（负信号）通过对比学习建模负向行为。

## 3 多模态&行为-wenqi 
* 本次：将item的多模态特征/语义特征融入到user2item的行为数据学习当中。
* 下次：1）探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。2）人力问题
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md) , [tencent doc](https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2)。
* 方案简介：GCN做推荐问题有两个：1）随机负采样，其实是不准确的。2）正样本很稀疏。解决办法：1）用正样本的自监督学习DINO（不同的跳与跳相似。可以作为自监督的信号），通过只构建正样本对进行对比学习[借鉴CV及nlp中的方法；2）使用样本的多模态特征，提升item的潜在表示-FREEDOM i2i。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
* 本次：和zhuoxi一起细化方案. recbole。 补一张图？
* 下次： @yuanfei强化人力问题
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
* 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用


# 8 meeting 20240507
## 1 社交&行为：
* 本次进展：在 LastFM 数据集上的结果表现不佳，加入对比损失反而会降低推荐的性能，下一步需要从以下几个方面分析新加入的对比损失对社交推荐的作用: 1) 稀疏 social graph 可能才会起作用; 2) t-SNE 可视化user embedding; 3) 用户分层; 4) @50; 5) 换数据集. 
* 下次预期：1）更换行为信号稀疏且社交信号稠密的数据集合，观察在recall@10 recall@20，recall@50 recall@100的表现。关于graph去噪声的参考资料：TransN: Heterogeneous Network Representation Learning by Translating Node Embeddings @wanglin
* 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
* 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。
  
## 2 搜索信号运用到推荐里面--qichao
* 本次: 
* 下次：qichao更新方案，祥源更新下自己思考到 doc上。@qichao @xingyuan
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md)
* 方案简介：搜索信号具体行为序列（正信号），与推荐里面曝光未点击行为序列（负信号）通过对比学习建模负向行为。

## 3 多模态&行为-wenqi 
* 本次：将item的多模态特征/语义特征融入到user2item的行为数据学习当中。
* 下次： 实现交错更新 @wenqi @chuchun @xinyu
* 方案简介：GCN做推荐问题有两个：1）随机负采样，其实是不准确的。2）正样本很稀疏。解决办法：1）用正样本的自监督学习DINO（不同的跳与跳相似。可以作为自监督的信号），通过只构建正样本对进行对比学习[借鉴CV及nlp中的方法；2）使用样本的多模态特征，提升item的潜在表示-FREEDOM i2i。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
* 本次：和zhuoxi一起细化方案. recbole。 补一张图？
* 下次： 着手实现RL代码 @pengfei @wenhao
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
* 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用



# 9 meeting 20240604
## 1 社交&行为：
* 本次进展：1）更换行为信号稀疏且社交信号稠密的数据集合，观察在recall@10 recall@20，recall@50 recall@100的表现。关于graph去噪声的参考资料：TransN: Heterogeneous Network Representation Learning by Translating Node Embeddings @wanglin 
* 下次预期：引入对比学习时，尝试不同的社交信号清洗方式，观察稀疏行为用户推荐效果的提升（多试几个不同的数据集）@wanglin @wangweisong
* 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
* 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。
  
## 2 搜索信号运用到推荐里面--qichao
* 本次: qichao更新方案，祥源更新下自己思考到 doc上。@qichao @xiangyuan，目前xiangyuan有初步的方案是把推荐和搜索的信息，进行对齐，然后在baidu进行实验
* 下次：
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md)
* 方案简介：搜索信号具体行为序列（正信号），与推荐里面曝光未点击行为序列（负信号）通过对比学习建模负向行为。

## 3 多模态&行为-wenqi 
* 本次： @wenqi @chuchun @xinyu
* 下次： 实现DRAGON和LayerGCN的结合
* 方案简介：
目前依然是考虑将item多模态信息融合进user-item的交互信息当中
考虑基于MMGCN和LayerGCN的idea
和以下论文中的模型进行结合:
MMSSL: Multi-Modal self-supervised Learning for Recommendation
DRAGON: Enhancing Dyadic Relations with Homogeneous Graphs for multimodal Recommendation
初步计划在DRAGON的模型上面去改，然后加入LayerGcN
数据暂时定为Amazon的Baby数据集，上面的两个模型相当于Baseline。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
* 本次：着手实现RL代码 @wenhao @pengfei  @liuweida
* 

# 8 meeting 20240702
## 1 社交&行为：
* 本次进展：引入对比学习时，尝试不同的社交信号清洗方式，观察稀疏行为用户推荐效果的提升（多试几个不同的数据集）@wanglin @wangweisong

lastfm数据集：

| 分位点   | 0    | 0.01 | 0.02  | 0.03 | 0.05 | 0.1  | 1    |
| -------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- |
| 交互次数 | 1    | 7.91 | 22.82 | 31   | 34   | 36   | 48   |

根据训练集中用户交互次数去筛选加入对比学习的用户，效果如下：

| Metric       | baseline（SocialLGN） | <8（0.01） | <23（0.02） | <31（0.03） | <36（0.1） |
| ------------ | --------------------- | ---------- | ----------- | ----------- | ---------- |
| Recall@10    | 19.72%                | 19.70%     | 19.71%      | 19.70%      | 19.57%     |
| NDCG@10      | 24.91%                | **24.93%** | 24.92%      | 24.91%      | 24.79%     |
| Precision@10 | 19.20%                | 19.18%     | 19.18%      | 19.17%      | 19.13%     |
| Recall@20    | 27.29%                | **27.41%** | **27.42%**  | **27.39%**  | **27.33%** |
| NDCG@20      | 27.46%                | **27.55%** | **27.54%**  | **27.51%**  | 27.43%     |
| Precision@20 | 13.38%                | **13.45%** | **13.45%**  | **13.43%**  | 13.39%     |

验证了socialnetwork图通过对比学习进行数据增强的有效性。

* 下次预期：寻找稀疏行为用户占比高的数据集，再次验证 @wanglin @wangweisong
* 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
* 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。
  
## 2 搜索信号运用到推荐里面--qichao
* 本次: qichao更新方案，祥源更新下自己思考到 doc上。@qichao @xiangyuan，目前xiangyuan有初步的方案是把推荐和搜索的信息，进行对齐，然后在baidu进行实验
* 下次： 细化下设计方案 @刘祥源  @冯智键 @宋瑞雪 
* 参考文档： [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md)
* 方案简介：搜索信号具体行为序列（正信号），与推荐里面曝光未点击行为序列（负信号）通过对比学习建模负向行为。
* 方案 @xiangyuan：
- 当前多数平台都提供了搜索和推荐服务，但是两个服务通常是采用不同的模型分别建模，没有考虑到用户在搜索与推荐两个表示空间中的关联关系，通过构建两者间的联系，能够进一步挖掘用户潜在意图及兴趣，在满足用户需求的基础上，进一步激发用户消费。当前构建联系考虑以下方案：
  1. 构建用户搜索、搜索点击和feed消费混合序列，双向建模序列元素关系和用户意图、偏好表示
  2. 从用户搜索序列和feed消费序列中根据序列相似性筛选相似用户，引入相似用户历史序列增强目标用户表示，拓展用户兴趣
* 参考资料：
- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)

## 3 多模态&行为-wenqi 
* 本次： @wenqi @chuchun @xinyu
* 下次： @艾长青-腾讯-搜推 和 @刘文奇+shopee 了解下之前的方案。
* 方案简介：
目前依然是考虑将item多模态信息融合进user-item的交互信息当中
考虑基于MMGCN和LayerGCN的idea
和以下论文中的模型进行结合:
MMSSL: Multi-Modal self-supervised Learning for Recommendation
DRAGON: Enhancing Dyadic Relations with Homogeneous Graphs for multimodal Recommendation
初步计划在DRAGON的模型上面去改，然后加入LayerGcN
数据暂时定为Amazon的Baby数据集，上面的两个模型相当于Baseline。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
* 本次：着手实现RL代码 @wenhao @pengfei  @liuweida
* 下次： 超空间方案，写论文@郑文豪 ，强化方案, 玄基跟@刘伟达 讲解之前的推荐算法+强化学习的方案。
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2
* 下次： 
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
* 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用
