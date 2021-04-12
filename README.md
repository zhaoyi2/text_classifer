# text_classifer
基于pytroch的文本分类相关算法实现
### model
- textcnn: Convolutional Neural Networks for Sentence Classification 

- HAN: Hierarchical Attention Networks for Document Classification

- DPCNN: Deep Pyramid Convolutional Neural Networks for Text Categorization

- Transformer: Attention Is All You Need,代码中实现了绝对和相对位置编码可选

- BERT:BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,开发中

### loss
- 包含softmax的改进版AM-softmax

### Dependencies
- pytroch==1.1.0
- torchtext

### experiment
在业务相关数据集上的表现:
- acc:bert最优，其次textCNN，HAN和DPCNN相对弱
- 性能:textCNN性能最优，4核16G上，平均响应时间2ms，HAN和DPCNN稍弱，BERT最差，单个响应耗时200ms+(不过，通过微软onnxruntime转化后，性能能够提升一倍)

