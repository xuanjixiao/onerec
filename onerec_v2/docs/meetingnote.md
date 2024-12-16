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
* 下次： 超空间方案，写论文@郑文豪 ，强化方案, 玄基跟@刘伟达 讲解之前的推荐算法+强化学习的方案。 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
* 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
* 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用



# 8 meeting 20240806

## 1 社交&行为：

- 本次进展：寻找稀疏行为用户占比高的数据集，再次验证 @wanglin @wangweisong
- 下次预期：进行两个数据集合完整实验，之后开始准备论文大纲

ciao数据集：

| 分位点   | 0    | 0.1  | 0.2  | 0.3  | 0.4  | 0.5  | 1    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 交互次数 | 2    | 6    | 7    | 9    | 11   | 15   | 1260 |

根据训练集中用户交互次数去筛选加入对比学习的用户，效果如下：

| Metric       | baseline（SocialLGN） | 9          | 10         | 11         | 20         | 30         | 40         |
| ------------ | --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Recall@10    | 0.0388                | **0.0392** | 0.0388     | **0.0389** | 0.0385     | 0.0383     | 0.0384     |
| NDCG@10      | 0.0417                | **0.0419** | **0.0419** | **0.0420** | **0.0418** | 0.0416     | 0.0415     |
| Precision@10 | 0.0255                | **0.0257** | **0.0256** | **0.0258** | 0.0255     | 0.0253     | 0.0255     |
| Recall@20    | 0.0572                | **0.0573** | 0.0572     | 0.0567     | 0.0569     | 0.0565     | 0.0557     |
| NDCG@20      | 0.0459                | **0.0461** | **0.0461** | **0.0461** | **0.0460** | 0.0458     | 0.0453     |
| Precision@20 | 0.0188                | **0.0191** | **0.0191** | **0.0190** | **0.0190** | **0.0189** | 0.0188     |
| MRR@10       | 1.1419                | **1.1444** | 1.1346     | **1.1477** | 1.1221     | 1.1079     | **1.1475** |
| AUC@10       | 0.0511                | 0.0494     | 0.0501     | 0.0508     | 0.0511     | 0.0500     | 0.0511     |
| MRR@20       | 2.9870                | **3.0579** | **3.0492** | **3.0227** | **3.0139** | **3.0062** | 2.9765     |
| AUC@20       | 0.1544                | 0.1526     | 0.1529     | 0.1529     | 0.1523     | 0.1512     | 0.1528     |

与上次结论一致。为了快速验证效果，这两次结果都是用大batchsize、大lr训出来的。目前正按SocialLGN原文的设定训练中。

- 下次预期：
- 参考文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
- 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。

## 2 搜索信号运用到推荐里面

- 下次预期：后续单独开会进行讨论，确定方案。ruixue方案给出一个示意图。

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

#### 参考资料：

- [UniSAR: Modeling User Transition Behaviors between Search and Recommendation](https://arxiv.org/abs/2404.09520)
- [When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation](https://arxiv.org/abs/2305.10822)
- [Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation](https://arxiv.org/pdf/2407.00912)

## 3 多模态&行为-wenqi 

- 本次： @wenqi @chuchun @xinyu
- 下次： 再调研增加两个创新点。
- 方案简介：
  目前依然是考虑将item多模态信息融合进user-item的交互信息当中
  考虑基于MMGCN和LayerGCN的idea
  和以下论文中的模型进行结合:
  MMSSL: Multi-Modal self-supervised Learning for Recommendation
  DRAGON: Enhancing Dyadic Relations with Homogeneous Graphs for multimodal Recommendation
  初步计划在DRAGON的模型上面去改，然后加入LayerGcN
  数据暂时定为Amazon的Baby数据集，上面的两个模型相当于Baseline。

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 

- 本次：着手实现RL代码 @wenhao @pengfei  @liuweida
- 下次： 1）超空间方案，确定论文写作时间。如果zhuoxi没时间，文豪和玄基可以先进行写作。可以先完成大纲挂在arxiv上，逐步修改。2）伟达跟文豪沟通下节奏，算法方案比较简单易于实现，看具体困难。
- 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
- 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
- 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用


# 8 meeting 20240902

## 1 社交&行为：
- 本次进展：王林验证效果，ciao数据集效果好，lastfm效果不好。 @wanglin @wangweisong
- 下次预期：1）lastfm调整学习率，和baseline一直为0.001查看效果。2）尝试玄基提出的四种view的方案 3）寻找新数据集合。
  玄基方案：1）我们之所以用社交信息，是认为社交关系紧密的用户，他们的物品偏好一样。那么为什么不直接试试朋友们喜欢的物品，作为我自己可能的兴趣描述呢（当然具体的单个物品可能是噪声）。我是担心我们没有直接学到这样一种二度信息传递:user通过社交关系找到了好友user，好友user通过行为关联到点击过的item。这样user就关联上了好友点击过的item。2）形式上，比较完备，四种对称的view。 3）问题是，user--user--item中，第二步user-item传递可能会有噪声或者不准确：朋友是随意点击的，或者我朋友喜欢的东西我不一定喜欢。解决方法是：1）随意点击：找到第二步user-item强关联，比如只用buy或者subscirbe信号，或者不使用点击使用点击率的预估分作为user-item的关联强度 2）我和朋友不一致：只有第一步user-user关系特别好的，才关联第二步。 具体方案见：[https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md)

- 文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
- 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。

## 2 搜索信号运用到推荐里面
- 本次进展：xiangyuan初步方案给出。
- 下次预期：xiangyuan把方案给到大家，和瑞雪，zhijian一起讨论新方案

#### 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md）

## 3 多模态&行为-changqing
- 本次： 设计了7个方法点，https://docs.qq.com/doc/DRW9WcnlkUEpDclhk，进行实验和讨论
- 下次： 后续调研，聚焦两个点：1）多模态fusion，2）构图时候topk近似改成无参数attention方案---这个需要找到代表的物理意义和优点。

#### 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md)

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
- 本次：liuweida实现baseline中
- 下次： 实现完baseline，和文豪沟通。
- 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
- 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
- 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用



# 8 meeting 20241010

## 1 社交&行为：
- 本次进展：王林验证效果，ciao数据集效果好，lastfm效果不好。 @wanglin @wangweisong
- 下次预期：1）lastfm调整学习率，和baseline一直为0.001查看效果。2）尝试玄基提出的四种view的方案 3）寻找新数据集合。
  玄基方案：1）我们之所以用社交信息，是认为社交关系紧密的用户，他们的物品偏好一样。那么为什么不直接试试朋友们喜欢的物品，作为我自己可能的兴趣描述呢（当然具体的单个物品可能是噪声）。我是担心我们没有直接学到这样一种二度信息传递:user通过社交关系找到了好友user，好友user通过行为关联到点击过的item。这样user就关联上了好友点击过的item。2）形式上，比较完备，四种对称的view。 3）问题是，user--user--item中，第二步user-item传递可能会有噪声或者不准确：朋友是随意点击的，或者我朋友喜欢的东西我不一定喜欢。解决方法是：1）随意点击：找到第二步user-item强关联，比如只用buy或者subscirbe信号，或者不使用点击使用点击率的预估分作为user-item的关联强度 2）我和朋友不一致：只有第一步user-user关系特别好的，才关联第二步。 具体方案见：[https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md)

- 文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
- 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。

## 2 搜索信号运用到推荐里面
- 本次进展：xiangyuan初步方案给出。
- 下次预期：xiangyuan把方案给到大家，和瑞雪，zhijian一起讨论新方案

#### 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md）

## 3 多模态&行为-changqing
- 本次： 设计了7个方法点，https://docs.qq.com/doc/DRW9WcnlkUEpDclhk，进行实验和讨论
- 下次： 后续调研，聚焦两个点：1）多模态fusion，2）构图时候topk近似改成无参数attention方案---这个需要找到代表的物理意义和优点。

#### 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md)

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
- 本次：liuweida实现baseline中
- 下次： 实现完baseline，和文豪沟通。
- 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
- 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
- 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用



# 8 meeting 20241105

## 1 社交&行为：
- 本次进展：
- 下次预期：写论文投稿ICDE，截止日期1125
- 文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
- 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。

## 2 搜索信号运用到推荐里面
- 本次进展：
- 下次预期：xiangyuan,和瑞雪，xuanji制定最终设计方案。
- 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md）

## 3 多模态&行为-changqing
- 本次：整体多模态fusion效果是0.096，但是baseline dragon是0.1021，https://docs.qq.com/doc/DRW9WcnlkUEpDclhk，
- 下次： 后续调研，聚焦两个点：1）多模态fusion 继续调整，2）大家分头做实验。
- 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md)

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
- 本次：1）整体样本上目前没有一个特别好的样本，当前样本可用，但是多个场景的多个样本的场景间次序没有标识。2）zhijian跑通RL代码，但是训练发现reward上升。
- 下次：继续调整样本和代码，保证模型实现正常训练。
- 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
- 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
- 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用


# 8 meeting 20241203

## 1 社交&行为-wanglin，wangweisong：
- 本次进展：论文投稿完成；
- 下次预期：1）补充interest alignment部分实验。之后考虑撰写长论文。
- 文档： 1） [readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_socia4rec.md), 2）[overleaf doc](https://www.overleaf.com/read/vnzvthkwdhdn#70e5f4)
- 方案简介：1）socialnetwork存在噪音和稀疏问题，我们使用svd方法进行去噪处理，然后得到的user embeding结果生成新的socialnetwrok图。新旧socialnetwork图通过contrastive learning方法学习，进行数据增强。2）对两个兴趣进行融合

## 2 搜索&推荐-
- 本次进展：
- 下次预期：xiangyuan,和瑞雪，xuanji制定最终设计方案。
- 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_search_based_rec.md）

## 3 多模态&行为-changqing,chucheng
- 本次：
- 下次： 重新梳理方案，按照extract-alignment&diversity-fusion方案制定大框架。
- 文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_modal.md)

## 4  multi-business-domain 跨业务域场景建模，直播，短视频，电商，社交，金融。-zhuoxi（hyperspace）/kexin（强化学习）/wenhao 
- 本次：1）模型正常训练
- 下次：继续调优效果。
- 目前方案：强化学习选择哪个场景是用户容易成交的场景，在多场景学习的时候增大它的样本的权重。
- 参考文档：[readme page](https://github.com/xuanjixiao/onerec/blob/onerecv2/onerec_v2/docs/onerecv2_multi_domain.md)
- 方案简介：1）使用域迁移，把domain1的u2u关系迁移到domain2，解决domain1的新用户问题。2）中间使用



