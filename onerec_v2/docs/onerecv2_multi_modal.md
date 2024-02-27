

目标：将item的多模态特征/语义特征融入到user2item的行为数据学习当中
方案：1）由于前任设计的多模态item2item图模型不太理想，故尝试设计一个高效的item2item的图学习方式。初步的方案可以使用行为数据学习的图模型作为辅助，交叉融合更新行为>学习图模型和多模态图模型。
      2）结合CV领域中自监督的方式，并结合图模型的特性，设计一个只用正样本学习的自监督学习方式，提高模型学习的效率
      3) 探索行为图模型和多模态图模型的融合方式，提高最终的效果
      4)相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2
方案：探索行为图模型和多模态图模型的融合方式，交错更新，单个epoch行为图学习多模态图学习对方信息，下个epoch反过来。相关文档：【腾讯文档】MMRec Insights https://docs.qq.com/doc/DSnR2c0lVTHBjbWx2。
