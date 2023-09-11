# ML-Reading-Lists

Decision Factors:
This list includes papers that have significantly shaped the field of ML, especially with the advent of deep learning techniques.
We fully aware that we might miss a paper or two but in a rapid changing industry we think these papaer will be sufficient to serve as the foundations for each field - youre more than welcome to offer changes - but our goal is to keep each reading list with a Max of 10 papers.

Just to clarify - this isn't the latest research papers of each papers!


## Table of Contents
- [NLP](#nlp)
- [Computer Vision](#computer-vision)
- [Generative Models](#generative-models)
- [Graph Neural Networks](#graph-neural-metworks)
- [Fairness in Machine Learning](#fairness-in-machine-learning)
- [Explainability in Machine Learning](#explainability-in-machine-learning)

## NLP

### 1. **Word Embeddings: Word2Vec**

- **Title**: Efficient Estimation of Word Representations in Vector Space
- **Authors**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- **Year**: 2013
- **Summary**: Introduced word2vec, revolutionizing word embeddings.
- **Link**: [arXiv](https://arxiv.org/abs/1301.3781)

### 2. **Sequence-to-Sequence Models**

- **Title**: Sequence to Sequence Learning with Neural Networks
- **Authors**: Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- **Year**: 2014
- **Summary**: Introduced the Seq2Seq model, foundational for machine translation and other tasks.
- **Link**: [arXiv](https://arxiv.org/abs/1409.3215)

### 3. **Attention Mechanism**

- **Title**: Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year**: 2014
- **Summary**: Introduced the attention mechanism, which has become crucial in NLP.
- **Link**: [arXiv](https://arxiv.org/abs/1409.0473)

### 4. **Transformer Models**

- **Title**: Attention Is All You Need
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
- **Year**: 2017
- **Summary**: Introduced the Transformer model, the foundation for many modern NLP models.
- **Link**: [arXiv](https://arxiv.org/abs/1706.03762)

### 5. **BERT**

- **Title**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year**: 2018
- **Summary**: Introduced BERT, which set new standards for several NLP tasks.
- **Link**: [arXiv](https://arxiv.org/abs/1810.04805)

### 6. **GPT (Generative Pre-trained Transformer)**

- **Title**: Improving Language Understanding by Generative Pre-Training
- **Authors**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- **Year**: 2018
- **Summary**: Introduced the GPT architecture, another milestone in language models.
- **Link**: [OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### 7. **ELMo (Embeddings from Language Models)**

- **Title**: Deep contextualized word representations
- **Authors**: Matthew E. Peters, Mark Neumann, Mohit Iyyer, et al.
- **Year**: 2018
- **Summary**: Introduced ELMo, showing the importance of contextualized word embeddings.
- **Link**: [arXiv](https://arxiv.org/abs/1802.05365)

### 8. **XLNet**

- **Title**: XLNet: Generalized Autoregressive Pretraining for Language Understanding
- **Authors**: Zhilin Yang, Zihang Dai, Yiming Yang, et al.
- **Year**: 2019
- **Summary**: Introduced XLNet, which outperformed BERT on several benchmarks.
- **Link**: [arXiv](https://arxiv.org/abs/1906.08237)

### 9. **RoBERTa**

- **Title**: RoBERTa: A Robustly Optimized BERT Pretraining Approach
- **Authors**: Yinhan Liu, Myle Ott, Naman Goyal, et al.
- **Year**: 2019
- **Summary**: Introduced RoBERTa, an optimized version of BERT.
- **Link**: [arXiv](https://arxiv.org/abs/1907.11692)

### 10. **T5 (Text-to-Text Transfer Transformer)**

- **Title**: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- **Authors**: Colin Raffel, Noam Shazeer, Adam Roberts, et al.
- **Year**: 2019
- **Summary**: Introduced T5, which reframed all NLP tasks as text-to-text tasks.
- **Link**: [arXiv](https://arxiv.org/abs/1910.10683)

## Computer Vision

### 1. **Convolutional Neural Networks (LeNet)**

- **Title**: Gradient-Based Learning Applied to Document Recognition
- **Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
- **Year**: 1998
- **Summary**: Introduced Convolutional Neural Networks (CNNs), setting the stage for deep learning in computer vision.
- **Link**: [Stanford](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

### 2. **ImageNet & AlexNet**

- **Title**: ImageNet Classification with Deep Convolutional Neural Networks
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Year**: 2012
- **Summary**: Described AlexNet, the CNN that significantly outperformed existing algorithms in the ImageNet competition.
- **Link**: [NIPS](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### 3. **VGGNet**

- **Title**: Very Deep Convolutional Networks for Large-Scale Image Recognition
- **Authors**: Karen Simonyan, Andrew Zisserman
- **Year**: 2014
- **Summary**: Introduced VGGNet, emphasizing the importance of depth in convolutional neural networks.
- **Link**: [arXiv](https://arxiv.org/abs/1409.1556)

### 4. **GoogLeNet/Inception**

- **Title**: Going Deeper with Convolutions
- **Authors**: Christian Szegedy, Wei Liu, Yangqing Jia, et al.
- **Year**: 2015
- **Summary**: Introduced the Inception architecture, which used "network-in-network" convolutions to increase efficiency.
- **Link**: [arXiv](https://arxiv.org/abs/1409.4842)

### 5. **Residual Networks (ResNet)**

- **Title**: Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **Summary**: Introduced residual learning, enabling the training of very deep networks.
- **Link**: [arXiv](https://arxiv.org/abs/1512.03385)

### 6. **YOLO (You Only Look Once)**

- **Title**: You Only Look Once: Unified, Real-Time Object Detection
- **Authors**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **Year**: 2016
- **Summary**: Introduced YOLO, a real-time object detection system.
- **Link**: [arXiv](https://arxiv.org/abs/1506.02640)

### 7. **U-Net: Image Segmentation**

- **Title**: U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Year**: 2015
- **Summary**: Introduced U-Net, a specialized network for semantic segmentation in biomedical image analysis.
- **Link**: [arXiv](https://arxiv.org/abs/1505.04597)

### 8. **Mask R-CNN**

- **Title**: Mask R-CNN
- **Authors**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
- **Year**: 2017
- **Summary**: Extended Faster R-CNN to provide pixel-level segmentation masks.
- **Link**: [arXiv](https://arxiv.org/abs/1703.06870)

### 9. **Capsule Networks**

- **Title**: Dynamic Routing Between Capsules
- **Authors**: Geoffrey E. Hinton, Alex Krizhevsky, Sida Wang
- **Year**: 2017
- **Summary**: Introduced capsule networks as an alternative to CNNs for hierarchical feature learning.
- **Link**: [arXiv](https://arxiv.org/abs/1710.09829)

### 10. **Neural Style Transfer**

- **Title**: A Neural Algorithm of Artistic Style
- **Authors**: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
- **Year**: 2015
- **Summary**: Introduced the concept of neural style transfer, using deep learning to transfer artistic styles between images.
- **Link**: [arXiv](https://arxiv.org/abs/1508.06576)

## Generative Models

### 1. **Generative Adversarial Networks**

- **Title**: Generative Adversarial Nets
- **Authors**: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **Year**: 2014
- **Summary**: Introduced GANs, a revolutionary framework for training generative models.
- **Link**: [arXiv](https://arxiv.org/abs/1406.2661)

### 2. **Variational Autoencoders (VAEs)**

- **Title**: Auto-Encoding Variational Bayes
- **Authors**: Diederik P. Kingma, Max Welling
- **Year**: 2013
- **Summary**: Introduced VAEs, offering a probabilistic approach to generating data.
- **Link**: [arXiv](https://arxiv.org/abs/1312.6114)

### 3. **Transformers for Text Generation (GPT)**

- **Title**: Improving Language Understanding by Generative Pre-Training
- **Authors**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- **Year**: 2018
- **Summary**: Introduced the GPT architecture, a milestone in text generation.
- **Link**: [OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### 4. **Bidirectional Transformers for Language Understanding (BERT)**

- **Title**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year**: 2018
- **Summary**: Introduced BERT, which has been adapted for various generative tasks.
- **Link**: [arXiv](https://arxiv.org/abs/1810.04805)

### 5. **CycleGAN**

- **Title**: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **Authors**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
- **Year**: 2017
- **Summary**: Introduced CycleGANs for image-to-image translation without paired data.
- **Link**: [arXiv](https://arxiv.org/abs/1703.10593)

### 6. **Style Transfer**

- **Title**: A Neural Algorithm of Artistic Style
- **Authors**: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
- **Year**: 2015
- **Summary**: Introduced the concept of neural style transfer, using deep learning to transfer artistic styles between images.
- **Link**: [arXiv](https://arxiv.org/abs/1508.06576)

### 7. **Normalizing Flows**

- **Title**: Variational Inference with Normalizing Flows
- **Authors**: Danilo Rezende, Shakir Mohamed
- **Year**: 2015
- **Summary**: Introduced Normalizing Flows for more flexible variational inference.
- **Link**: [arXiv](https://arxiv.org/abs/1505.05770)

### 8. **PixelRNN**

- **Title**: Pixel Recurrent Neural Networks
- **Authors**: Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
- **Year**: 2016
- **Summary**: Introduced PixelRNNs, a model for generating images pixel by pixel.
- **Link**: [arXiv](https://arxiv.org/abs/1601.06759)

### 9. **Wasserstein GAN**

- **Title**: Wasserstein GAN
- **Authors**: Martin Arjovsky, Soumith Chintala, Léon Bottou
- **Year**: 2017
- **Summary**: Introduced the Wasserstein loss for more stable GAN training.
- **Link**: [arXiv](https://arxiv.org/abs/1701.07875)

### 10. **BigGAN**

- **Title**: Large Scale GAN Training for High Fidelity Natural Image Synthesis
- **Authors**: Andrew Brock, Jeff Donahue, Karen Simonyan
- **Year**: 2018
- **Summary**: Discussed scaling up GANs to generate high-quality images.
- **Link**: [arXiv](https://arxiv.org/abs/1809.11096)

## Graph Neural Networks

### 1. **Spectral Networks and Locally Connected Networks on Graphs**

- **Title**: Spectral Networks and Locally Connected Networks on Graphs
- **Authors**: Joan Bruna, Wojciech Zaremba, Arthur Szlam, Yann LeCun
- **Year**: 2013
- **Summary**: One of the earliest works on graph neural networks, introducing the concept of spectral networks.
- **Link**: [arXiv](https://arxiv.org/abs/1312.6203)

### 2. **Graph Convolutional Networks (GCNs)**

- **Title**: Semi-Supervised Classification with Graph Convolutional Networks
- **Authors**: Thomas N. Kipf, Max Welling
- **Year**: 2016
- **Summary**: Introduced Graph Convolutional Networks, a fundamental architecture for GNNs.
- **Link**: [arXiv](https://arxiv.org/abs/1609.02907)

### 3. **GraphSAGE**

- **Title**: Inductive Representation Learning on Large Graphs
- **Authors**: William L. Hamilton, Rex Ying, Jure Leskovec
- **Year**: 2017
- **Summary**: Introduced GraphSAGE, a method for inductive learning on graphs.
- **Link**: [arXiv](https://arxiv.org/abs/1706.02216)

### 4. **GAT (Graph Attention Networks)**

- **Title**: Graph Attention Networks
- **Authors**: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
- **Year**: 2017
- **Summary**: Introduced Graph Attention Networks, integrating attention mechanisms into GNNs.
- **Link**: [arXiv](https://arxiv.org/abs/1710.10903)

### 5. **Graph Neural Networks with Differentiable Pooling**

- **Title**: Hierarchical Graph Representation Learning with Differentiable Pooling
- **Authors**: Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec
- **Year**: 2018
- **Summary**: Introduced differentiable pooling layers for learning hierarchical representations of graphs.
- **Link**: [arXiv](https://arxiv.org/abs/1806.08804)

### 6. **ChebNet**

- **Title**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
- **Authors**: Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst
- **Year**: 2016
- **Summary**: Introduced ChebNet, which uses Chebyshev polynomials for spectral graph convolutions.
- **Link**: [arXiv](https://arxiv.org/abs/1606.09375)

### 7. **Graph Isomorphism Networks (GIN)**

- **Title**: How Powerful are Graph Neural Networks?
- **Authors**: Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
- **Year**: 2018
- **Summary**: Investigated the expressive power of GNNs and introduced Graph Isomorphism Networks.
- **Link**: [arXiv](https://arxiv.org/abs/1810.00826)

### 8. **Message Passing Neural Network (MPNN)**

- **Title**: Neural Message Passing for Quantum Chemistry
- **Authors**: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
- **Year**: 2017
- **Summary**: Introduced the Message Passing Neural Network, a framework for learning on graphs.
- **Link**: [arXiv](https://arxiv.org/abs/1704.01212)

### 9. **Dynamic Graph CNN for Learning on Point Clouds**

- **Title**: Dynamic Graph CNN for Learning on Point Clouds
- **Authors**: Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon
- **Year**: 2018
- **Summary**: Extended GNNs to unstructured point clouds, often used in 3D vision tasks.
- **Link**: [arXiv](https://arxiv.org/abs/1801.07829)

### 10. **Relational Graph Convolutional Networks**

- **Title**: Modeling Relational Data with Graph Convolutional Networks
- **Authors**: Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling
- **Year**: 2017
- **Summary**: Extended GCNs to relational data, which is particularly useful for knowledge graphs.
- **Link**: [arXiv](https://arxiv.org/abs/1703.06103)

## Fairness in Machine Learning

### **1. Fairness Definitions Explained**

- **Title**: Fairness Definitions Explained
- **Authors**: Sahil Verma, Julia Rubin
- **Year**: 2018
- **Summary**: Provides a comprehensive overview of various fairness definitions in machine learning.
- **Link**: [Umass](https://fairware.cs.umass.edu/papers/Verma.pdf)

### **2. Fairness Through Awareness**

- **Title**: Fairness Through Awareness
- **Authors**: Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel
- **Year**: 2011
- **Summary**: Introduced the concept of "individual fairness."
- **Link**: [arXiv](https://arxiv.org/abs/1104.3913)

### **3. Equality of Opportunity in Supervised Learning**

- **Title**: Equality of Opportunity in Supervised Learning
- **Authors**: Moritz Hardt, Eric Price, Nathan Srebro
- **Year**: 2016
- **Summary**: Introduces the notion of equality of opportunity in the context of classification.
- **Link**: [arXiv](https://arxiv.org/abs/1610.02413)

## Explainability in Machine Learning

### **1. Local Interpretable Model-agnostic Explanations (LIME)**

- **Title**: "Why Should I Trust You?” Explaining the Predictions of Any Classifier
- **Authors**: Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
- **Year**: 2016
- **Summary**: Introduced LIME, a framework for explaining individual predictions.
- **Link**: [arXiv](https://arxiv.org/abs/1602.04938)

### **2. SHAP (SHapley Additive exPlanations)**

- **Title**: A Unified Approach to Interpreting Model Predictions
- **Authors**: Scott Lundberg, Su-In Lee
- **Year**: 2017
- **Summary**: Introduced SHAP values based on game theory for model explanation.
- **Link**: [arXiv](https://arxiv.org/abs/1705.07874)

### **3. Interpretable Decision Sets**

- **Title**: Interpretable Decision Sets: A Joint Framework for Description and Prediction
- **Authors**: Himabindu Lakkaraju, Stephen H. Bach, Jure Leskovec
- **Year**: 2016
- **Summary**: Focuses on generating interpretable decision sets for classification.
- **Link**:[Stanford](https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf)

### **4. Anchors: High-Precision Model-Agnostic Explanations**

- **Title**: Anchors: High-Precision Model-Agnostic Explanations
- **Authors**: Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
- **Year**: 2018
- **Summary**: Proposes a method for creating "anchor" explanations that are locally sufficient conditions for predictions.
- **Link**: [Washington](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)

### **5. Counterfactual Explanations**

- **Title**: Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR
- **Authors**: Sandra Wachter, Brent Mittelstadt, Chris Russell
- **Year**: 2017
- **Summary**: Discusses counterfactual explanations in the context of GDPR.
- **Link**: [arXiv](https://arxiv.org/abs/1711.00399)

### **6. Explainability for Neural Networks**

- **Title**: Towards A Rigorous Science of Interpretable Machine Learning
- **Authors**: Finale Doshi-Velez, Been Kim
- **Year**: 2017
- **Summary**: Discusses the challenges and directions for making neural networks interpretable.
- **Link**: [arXiv](https://arxiv.org/abs/1702.08608)

### **7. Towards Fairness in Visual Recognition**

- **Title**: Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation
- **Authors**: Zeyu Wang, Klint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, Olga Russakovsky
- **Year**: 2019
- **Summary**: Discusses fairness issues in computer vision and proposes bias mitigation strategies.
- **Link**: [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Towards_Fairness_in_Visual_Recognition_Effective_Strategies_for_Bias_Mitigation_CVPR_2020_paper.html)
