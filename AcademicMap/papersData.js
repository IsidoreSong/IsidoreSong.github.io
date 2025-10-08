// Formulas have been converted to valid LaTeX format
const face = [
  // 第一部分：基于Metric Learning的身份监督方法 (Classification-Based / Identity-Supervised Methods)
  
  // A. Core Models & Architectures：通过大规模Metric Learning学习特征
  {
    id: "deepid",
    title: "Deep Learning Face Representation from Predicting 10,000 Classes",
    authors: "Sun et al.",
    year: 2014,
    method: "Core Models & Architectures",
    category: "Metric Learning",
    influences: [],
    core_idea: "将人脸识别视为一个大规模（10,000 类）Metric Learning任务，使用 Softmax 损失进行端到端训练，隐式学习判别性特征表示，无需额外的Metric Learning损失。",
    formula: "\\mathcal{L} = -\\log \\left( \\frac{e^{f(x_i)_{y_i}}}{\\sum_{j=1}^{C} e^{f(x_i)_j}} \\right)",
    contribution: "开创性地验证了深度学习在人脸识别上的巨大潜力，提出“通过Metric Learning任务隐式学习判别性特征”的范式，成为后续研究的基础。"
  },
  {
    id: "deepid2",
    title: "Deep Learning Face Representation by Joint Identification-Verification",
    authors: "Sun et al.",
    year: 2014,
    method: "DeepID2",
    category: "Metric Learning",
    influences: ["deepid"],
    core_idea: "在 DeepID 基础上，融合识别损失（Softmax）与验证损失（对比损失，Contrastive Loss），实现双重优化：类内聚合 + 类间分离。",
    formula: "\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2N} \\sum_{i=1}^{N} \\left[ y_i \\cdot d_i^2 + (1 - y_i) \\cdot \\max(0, m - d_i)^2 \\right]",
    contribution: "第一次将「识别」与「验证」双重任务联合训练，是早期将Metric Learning与Metric Learning学习结合的典范，显著提升了跨姿态与光照下的识别鲁棒性。"
  },
  {
    id: "deepid2plus",
    title: "Deeply learned face representations are sparse, selective, and robust",
    authors: "Sun et al.",
    year: 2015,
    method: "DeepID2+",
    category: "Metric Learning",
    influences: ["deepid2"],
    core_idea: "在DeepID2基础上，进一步在中间层特征图上引入监督信号，使局部特征（CNN激活）也携带身份信息，增强特征的稀疏性与选择性。",
    formula: "\\mathcal{L} = \\mathcal{L}_{\\text{softmax}}^{final} + \\sum_{l=1}^{L} \\lambda_l \\mathcal{L}_{\\text{softmax}}^{l}",
    contribution: "提出“深度监督”机制，使网络每一层都能积极参与身份判别，极大增强了模型的鲁棒性与泛化能力，成为后续多尺度特征融合的模板。"
  },
  
  // B. 核心演进：角度与余弦间隔损失 (Margin-Based Losses)
  {
    id: "centerloss",
    title: "A Discriminative Feature Learning Approach for Deep Face Recognition",
    authors: "Wen et al.",
    year: 2016,
    method: "Center Loss",
    category: "Metric Learning",
    influences: ["deepid"],
    core_idea: "在传统 Softmax 损失之外，引入一个额外的“中心损失”（Center Loss），显式约束同一类别的特征向量靠近其类中心，从而增强类内紧凑性，同时保留 Softmax 的类间分离能力。",
    formula: "\\mathcal{L}_{\\text{center}} = \\frac{1}{2} \\sum_{i=1}^{N} \\| x_i - c_{y_i} \\|^2",
    contribution: "首次显式建模类内紧凑性，证明仅靠 Softmax 无法有效控制类内方差，为后续Metric Learning学习与角度损失铺平道路。"
  },
  {
    id: "normface",
    title: "NormFace & L2-constrained Softmax Loss for Discriminative Face Verification",
    authors: "Wang et al.",
    year: 2017,
    method: "L2-Softmax",
    category: "Metric Learning",
    influences: ["centerloss"],
    core_idea: "提出将特征和类别权重同时进行 L2 归一化，将所有特征“投影”到单位超球面上。这使得优化目标从同时考虑特征的“模长+角度”简化为“仅优化角度”，Metric Learning决策完全由余弦相似度（即夹角）决定，从而消除了特征模长对Metric Learning的干扰。",
    formula: "\\mathcal{L}_{\\text{L2-Softmax}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos\\theta_{y_i}}}{\\sum_{j=1}^{C} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "为后续所有角度损失（ArcFace、CosFace）奠定了理论和几何基础，阐明了“特征归一化+Metric Learning学习”是提升人脸识别性能的关键路径，确立了“超球面嵌入”的标准范式。"
  },
  {
    id: "lsoftmax",
    title: "Large-Margin Softmax Loss for Convolutional Neural Networks",
    authors: "Liu et al.",
    year: 2016,
    method: "L-Softmax",
    category: "Metric Learning",
    influences: ["deepid"],
    core_idea: "首次引入角度间隔概念，通过修改 Softmax 中的角度函数，强制类中心之间的最小夹角大于某个阈值，从而提升特征分布的分离度。",
    formula: "\\mathcal{L}_{\\text{L-Softmax}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\psi(\\theta_{y_i})}}{e^{s \\cdot \\psi(\\theta_{y_i})} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "首次在Softmax框架下引入人为的角度间隔，启发了后续 SphereFace、ArcFace 和 CosFace 的设计，是角度损失谱系的开创性工作。"
  },
  {
    id: "sphereface",
    title: "SphereFace: Deep Hypersphere Embedding for Face Recognition",
    authors: "Liu et al.",
    year: 2017,
    method: "SphereFace",
    category: "Metric Learning",
    influences: ["lsoftmax", "normface"],
    core_idea: "在 L-Softmax 的基础上，提出“乘性角度间隔”（multiplicative angular margin），将类别中心与样本特征之间的夹角放大 m 倍（m > 1），从而在超球面上强制形成更宽的Metric Learning边界，显著增强类间分离。",
    formula: "\\mathcal{L}_{\\text{SphereFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(m \\theta_{y_i})}}{e^{s \\cdot \\cos(m \\theta_{y_i})} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "首次证明乘性角度间隔比加性间隔在几何上更具判别力，是首个在超球面上实现强分开的损失函数，为 ArcFace 和 CosFace 提供了直接启发。"
  },
  {
    id: "cosface",
    title: "CosFace: Large Margin Cosine Loss for Deep Face Recognition",
    authors: "Wang et al.",
    year: 2018,
    method: "CosFace",
    category: "Metric Learning",
    influences: ["sphereface"],
    core_idea: "与 ArcFace 类似，但在余弦相似度空间（而非角度空间）中增加一个固定的“加性余弦间隔”（additive cosine margin），直接从相似度得分中减去一个惩罚项，以增大决策边界。",
    formula: "\\mathcal{L}_{\\text{CosFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot (\\cos\\theta_{y_i} - m)}}{e^{s \\cdot (\\cos\\theta_{y_i} - m)} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "提供了一种更简洁、数值更稳定的边界优化方式，在保持高性能的同时降低训练难度，常用于推理速度敏感的场景。"
  },
  {
    id: "arcface",
    title: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
    authors: "Deng et al.",
    year: 2019,
    method: "ArcFace",
    category: "Metric Learning",
    influences: ["cosface", "sphereface", "centerloss"],
    core_idea: "在归一化后的单位超球面上，对 Softmax 的决策边界施加“加性角度间隔”（additive angular margin），直接在角度空间中优化特征与类中心的夹角，即用 (θ_{y_i} + m) 替换 θ_{y_i}，实现最强的类间分离和类内聚集。",
    formula: "\\mathcal{L}_{\\text{ArcFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m)}}{e^{s \\cdot \\cos(\\theta_{y_i} + m)} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "目前工业界和学术界的主流方法，理论优美，性能卓越，成为人脸识别领域的“黄金标准”，被广泛用于人脸验证、活体检测等系统中。"
  },
  {
    id: "ringloss",
    title: "Ring Loss: Convex Feature Normalization for Face Recognition",
    authors: "Zhao et al.",
    year: 2019,
    method: "Ring Loss",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "通过一个额外的特征模长正则项，强制所有样本的特征向量（L2归一化后）收敛到一个固定半径的“球面”上，从而避免特征模长漂移导致的模型不稳定。",
    formula: "\\mathcal{L}_{\\text{Ring}} = \\frac{1}{N} \\sum_{i=1}^{N} (\\|x_i\\| - r)^2",
    contribution: "解决了归一化后模长仍可能发散的问题（尤其在跨数据集训练中），使得超球面嵌入更具几何一致性，在跨域人脸识别任务中表现突出。"
  },
  
  // C. 动态与自适应间隔 (Dynamic & Adaptive Margins)
  {
    id: "adaface",
    title: "AdaFace: Quality Adaptive Margin for Face Recognition",
    authors: "Kim et al.",
    year: 2023,
    method: "AdaFace",
    category: "Metric Learning",
    influences: ["magface"],
    core_idea: "提出一种感知质量的角度间隔机制：根据样本特征的模长（作为图像质量的代理指标）自适应调整间隔 m。对高质量图像（模长较大）使用更大的间隔进行严格约束，对低质量图像（模长较小）使用更小的间隔以放宽约束，从而提升整体泛化能力。",
    formula: "\\mathcal{L}_{\\text{AdaFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m(\\|x_i\\|))}}{e^{s \\cdot \\cos(\\theta_{y_i} + m(\\|x_i\\|))} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "首次实现“样本质量感知”的损失设计，解决了因姿态、模糊、光照导致的“信息不足样本被过度惩罚”问题，大幅提升了在真实场景（如监控、移动端）中的鲁棒性。"
  },
  {
    id: "magface",
    title: "MagFace: A Universal Representation for Face Recognition and Quality Assessment",
    authors: "Meng et al.",
    year: 2021,
    method: "MagFace",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "建立“特征模长 ↔ 图像质量”的正相关关系：高质量图像应具有更大的模长，低质量图像模长远小于平均水平，并据此设计一个与模长联动的自适应角度间隔。",
    formula: "\\mathcal{L}_{\\text{MagFace}} = \\mathcal{L}_{\\text{ArcFace}} + \\lambda_1 \\mathcal{L}_{\\text{regularization}} + \\lambda_2 \\mathcal{L}_{\\text{margin}}",
    contribution: "首次将“特征模长”与“图像质量”从经验挂钩上升为可学习的理论映射，实现了人脸识别与质量评估的双任务联合优化，成为工业级系统的首选。"
  },
  {
    id: "qmagface",
    title: "QMagFace: Simple and Accurate Quality-Aware Face Recognition",
    authors: "Peng et al.",
    year: 2023,
    method: "QMagFace",
    category: "Metric Learning",
    influences: ["magface", "adaface"],
    core_idea: "在 MagFace 的基础上，抛弃对模长的隐式建模，直接引入外部图像质量评分（如清晰度、遮挡、光照评估分数 Q），并将其作为角度间隔的直接输入，实现更精准的质量感知。",
    formula: "\\mathcal{L}_{\\text{QMagFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m_0 + \\alpha \\cdot (Q_i - \\bar{Q}))}}{e^{s \\cdot \\cos(\\theta_{y_i} + m_0 + \\alpha \\cdot (Q_i - \\bar{Q}))} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "证明知识驱动（非数据驱动）的质量信号可以更高效地提升性能，实现“无需重新设计网络结构、只需调整损失”的最优迁移能力，极大降低部署成本。"
  },
  {
    id: "fairloss",
    title: "Fair Loss: Margin-Aware Reinforcement Learning for Deep Face Recognition",
    authors: "Zhang et al.",
    year: 2022,
    method: "Fair Loss",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "摒弃全局统一的间隔，将每个类别视为一个独立的“决策代理”，通过强化学习动态调整其最优角度间隔，使少数类或困难类获得更多“训练资源”。",
    formula: "\\mathcal{L}_{\\text{Fair}} = \\mathbb{E}_{c \\sim \\mathcal{C}} \\left[ \\mathbb{E}_{x_i \\in \\mathcal{D}_c} \\left[ r_c(\\mathcal{C}) \\cdot \\log \\pi_{\\theta}(m_c | x_i) \\cdot \\mathcal{L}_{\\text{ArcFace}}(m_c) \\right] \\right]",
    contribution: "首次提出按“类”而非“样本”做自适应间隔，破解了长尾分布下“多数类主导训练”的问题，在大规模身份数据集（如 MS1M-v3）上显著提升尾部类识别率。"
  },
  {
    id: "mvsoftmax",
    title: "Mis-Classified Vector Guided Softmax Loss for Face Recognition",
    authors: "Liu et al.",
    year: 2021,
    method: "MV-Softmax",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "不是单纯拉近正类中心或推开负类，而是主动利用被错误Metric Learning样本的特征向量与它被误分到的类别中心之间的方向信息，来指导如何更精准地调整其与真实类中心的距离。",
    formula: "\\mathcal{L}_{\\text{MV}} = \\mathcal{L}_{\\text{ArcFace}} + \\lambda \\cdot \\| \\frac{v_i}{\\|v_i\\|} \\cdot (c_{y_i} - x_i) \\|",
    contribution: "将“判别性学习”从被动边界扩张，升级为主动错误引导修正，是首个基于错误样本反馈机制的损失函数，对噪声标签和边界混淆样本有显著抑制作用。"
  },
  {
    id: "scfarcface",
    title: "Spherical Confidence Learning for Face Recognition",
    authors: "Chen et al.",
    year: 2023,
    method: "SCF-ArcFace",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "将模型对样本Metric Learning的“置信度”（即当前预测概率）作为调整角度间隔的信号：对于低置信度样本（即模型不确定的难样本），施加更大间隔以强化学习；对于高置信度样本减小约束，避免过度拟合。",
    formula: "\\mathcal{L}_{\\text{SCF}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m_0 \\cdot \\exp(-k \\cdot p_{y_i}))}}{e^{s \\cdot \\cos(\\theta_{y_i} + m_0 \\cdot \\exp(-k \\cdot p_{y_i}))} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "首次将“深度模型的预测置信度”作为损失函数调节信号，实现训练过程的动态聚焦，使网络优先攻克“不确定但重要”的样本，接近人类学习中的“聚焦弱点”机制。"
  },
  
  // D. 训练策略与特定场景优化
  {
    id: "broadface",
    title: "BroadFace: Looking at Tens of thousands of People at once for Face Recognition",
    authors: "Liu et al.",
    year: 2021,
    method: "BroadFace",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "在传统 softmax 中，每个样本仅与一个正类和少量负类比较，导致训练效率低下。BroadFace 提出在每次迭代中为每个样本动态采样数百至数千个负类中心（而非仅用批次内的负样本），从而让模型在每一次更新中接触几乎完整的身份空间。",
    formula: "\\mathcal{L}_{\\text{BroadFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i})}}{e^{s \\cdot \\cos(\\theta_{y_i})} + \\sum_{k=1}^{K} e^{s \\cdot \\cos(\\theta_k^{\\text{neg}})}}",
    contribution: "首次证明扩大负样本空间的广度比单纯设计更复杂的间隔函数更能提升特征判别力。该方法使模型在训练初期就能“看到”整个身份世界的分布结构，显著加快收敛速度与泛化能力，成为大规模身份学习（如 MS1M-v3）的训练标准。"
  },
  {
    id: "curricularface",
    title: "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition",
    authors: "Huang et al.",
    year: 2020,
    method: "CurricularFace",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "人类学习遵循“由易到难”的曲线。CurricularFace 模仿这一过程，根据每个样本当前的Metric Learning难度（由其与目标类中心的余弦相似度 cosθ_i 与平均相似度 m̄ 的差值衡量），动态调整其在损失函数中的权重，使模型先学习简单样本，再逐步接触难样本。",
    formula: "\\mathcal{L}_{\\text{CurricularFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} w_i \\cdot \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m)}}{e^{s \\cdot \\cos(\\theta_{y_i} + m)} + \\sum_{j \\ne y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "提出“难度感知的课程学习”机制，避免模型早期被困难样本主导而陷入局部最优。该方法显著降低训练震荡，提高收敛稳定性，在 MegaFace、LFW 等基准上刷新当时 SOTA。"
  },
  {
    id: "adacos",
    title: "AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations",
    authors: "Zhong et al.",
    year: 2019,
    method: "AdaCos",
    category: "Metric Learning",
    influences: ["cosface"],
    core_idea: "传统方法中，余弦尺度因子 s 需人工调参（如 64），该参数影响梯度大小与收敛速度，且在不同数据集上表现不一致。AdaCos 令 s 自适应地根据批次统计量动态调整，无需手动设置。",
    formula: "s = \\frac{\\ln(C - 1)}{\\Delta}, \\quad \\text{其中} \\quad \\Delta = \\frac{1}{N} \\sum_{i=1}^{N} \\left( \\cos\\theta_{y_i} - \\bar{c} \\right)^2",
    contribution: "首次实现尺度超参数的无监督自适应，使损失函数具备更强的泛化能力，无需针对每个新数据集重新调参，极大提升工程部署便利性，成为许多后续工作的默认选用基准。"
  },
  {
    id: "p2sgrad",
    title: "P2SGrad: Refined Gradients for Optimizing Deep Face Models",
    authors: "Wang et al.",
    year: 2023,
    method: "P2SGrad",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "Vectorized 条件下的角度损失（如 ArcFace）在训练后期容易出现“梯度饱和”或“梯度爆炸”，尤其在难样本附近。P2SGrad 并不改变损失函数本身，而是重新设计梯度传递过程，构造更平滑、可控的更新方向。",
    formula: "\\nabla_{\\theta_i} \\mathcal{L} = \\gamma_i \\cdot \\nabla_{\\theta_i} \\mathcal{L}_{\\text{ArcFace}}",
    contribution: "将优化器从“损失驱动”升级为“梯度流驱动”，解决了角度损失在后期训练中收敛不稳定的问题，加速模型达到泛化最优解，已在 PyTorch 社区开源并被广泛采用。"
  },
  
  // 特定场景解决方案
  {
    id: "qgface",
    title: "QGFace: Quality-Guided Joint Training For Mixed-Quality Face Recognition",
    authors: "Song et al.",
    year: 2023,
    method: "QGFace",
    category: "Metric Learning",
    influences: ["arcface", "li2023blip2"],
    core_idea: "真实世界人脸图像质量差异巨大（从高清证件照到模糊监控帧），直接训练会导致模型偏向高质量数据。QGFace 提出联合两个损失——样本-类别损失（ArcFace）与样本-样本损失（对比损失）——并用图像质量分数 Q 动态加权二者。",
    formula: "\\mathcal{L}_{\\text{QGFace}} = (1 - Q_i) \\cdot \\mathcal{L}_{\\text{contrastive}} + Q_i \\cdot \\mathcal{L}_{\\text{ArcFace}}",
    contribution: "首次提出基于质量的双损失调度机制，在无高清标注的前提下实现异质数据的协同训练，大幅提升在安防、跨境识别场景下的实用性。"
  },
  {
    id: "coupleddiscriminative",
    title: "Coupled discriminative manifold alignment for low-resolution face recognition",
    authors: "Yang et al.",
    year: 2019,
    method: "Coupled Discriminative Manifold Alignment",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "低分辨率（LR）和高分辨率（HR）人脸属于不同流形，传统方法直接映射会导致语义失真。本文提出双流形对齐：在特征空间中，强制 LR 与 HR 图像的特征分布围绕同一身份中心聚集，并最小化两类流形间的几何距离。",
    formula: "\\text{包含识别损失、对比损失和流形正则化}",
    contribution: "首次将“分布对齐”作为低分辨率识别的主线，而非简单超分或特征映射，显著提升跨分辨率识别的泛化能力，奠定后续跨域人脸识别基础。"
  },
  {
    id: "tcn",
    title: "TCN: Transferable Coupled Network for Cross-Resolution Face Recognition",
    authors: "Wang et al.",
    year: 2020,
    method: "TCN",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "与其分别学习 HR 与 LR 的特征，不如构建一个“耦合特征空间”，使得来自不同分辨率的样本能直接比较。TCN 学习一个跨分辨率的联合嵌入函数。",
    formula: "\\text{同人：}\\| f_{\\text{HR}}(x) - f_{\\text{LR}}(x') \\|^2 \\leq \\delta; \\text{不同人：}\\| f_{\\text{HR}}(x) - f_{\\text{LR}}(y) \\|^2 \\geq \\delta + m",
    contribution: "提出“跨域耦合嵌入”范式，将问题从“HR→LR 映射”转变为“统一Metric Learning学习”，实现真正的跨分辨率人脸识别，成为工业级视频监控系统的基础组件。"
  },
  {
    id: "ideanet",
    title: "IDEA-Net: An Implicit Identity-Extended Data Augmentation for Low-Resolution Face Representation Learning",
    authors: "Chen et al.",
    year: 2022,
    method: "IDEA-Net",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "传统数据增强仅在像素空间添加噪声或旋转，无法解决“样本稀少”问题。IDEA-Net 在特征空间直接生成与原样本同身份但多样化的“虚拟特征”，扩大类内分布，缓解低分辨率下样本匮乏问题。",
    formula: "\\text{使用GAN生成特征向量 }\\tilde{x} = G(x, z)",
    contribution: "开创“特征级数据增强”新范式，突破域外数据收集限制，在仅靠少量低分辨率图像训练时依然具备强判别能力，特别适用于跨境、军事等数据受限场景。"
  },
  {
    id: "ddfm",
    title: "Deep Discriminative Feature Models (DDFMs) for Set Based Face Recognition",
    authors: "Zhang et al.",
    year: 2018,
    method: "DDFM",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "传统方法处理单张图像，而集合人脸识别（如监控视频中的多帧人脸）需建模图像集合内部的多样性与集合间的分离性。DDCFM 同时学习“判别性表示”和“集合间距离Metric Learning”。",
    formula: "d(\\mathcal{S}_i, \\mathcal{S}_j) = \\| \\phi(\\mathcal{S}_i) - \\phi(\\mathcal{S}_j) \\|",
    contribution: "首次建立“集合-集合而非单样本-单样本”的识别框架，适用于视频监控、空中人脸识别等实际应用，为序列式人脸识别提供理论基础。"
  },
  {
    id: "headposesoftmax",
    title: "HeadPose-Softmax: Head Pose Adaptive Curriculum Learning Loss for Deep Face Recognition",
    authors: "Zhou et al.",
    year: 2021,
    method: "HeadPose-Softmax",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "偏转角度（如侧脸）是导致识别困难的主要来源。本工作依据输入图像的头部姿态估计值（yaw/pitch）作为课程难度指标，低姿态（正面）样本优先学习，高姿态样本延迟权重增加。",
    formula: "\\mathcal{L}_{\\text{HeadPose-Softmax}} = -\\frac{1}{N} \\sum_{i=1}^{N} w_i \\cdot \\mathcal{L}_{\\text{ArcFace}}",
    contribution: "将物理先验（姿态）与训练动态结合，实现语义级课程学习，在多人脸数据库（如 CACD、MegaFace）上显著提升大姿态识别准确率，避免模型“只认正脸”。"
  },
  {
    id: "superidentity",
    title: "Super-Identity Convolutional Neural Network for Face Hallucination",
    authors: "Zhou et al.",
    year: 2021,
    method: "Super-Identity CNN",
    category: "Metric Learning",
    influences: ["arcface"],
    core_idea: "在超分辨率任务中，传统方法仅恢复像素细节，忽略身份一致性。该文提出“超身份损失”：利用目标身份的多张高分辨率图像，构建一个身份原型中心 c_i，并要求生成的高清图像特征必须靠近该中心，从而保证身份不变性。",
    formula: "\\mathcal{L}_{\\text{super-id}} = \\| f_{\\text{SR}}(x_{\\text{LR}}) - c_i \\|^2",
    contribution: "将人脸识别中的“样本-类别关系”作为生成模型的约束工具，使超分辨率不仅是“变清晰”，更是“变对人”，在公安图像恢复、历史人物重建等领域具有重要价值。"
  },
  
  // 第二部分：基于Metric Learning学习的对比方法 (Metric Learning-Based / Contrastive Methods)
  
  // A. 监督对比学习 (Supervised Contrastive Learning)
  {
    id: "contrastiveloss",
    title: "Dimensionality Reduction by Learning an Invariant Mapping",
    authors: "Hadsell et al.",
    year: 2006,
    method: "Contrastive Loss",
    category: "Metric Learning",
    influences: [],
    core_idea: "首次提出“对比损失”（Contrastive Loss）框架，为深度Metric Learning学习奠定基石。其目标不是Metric Learning，而是学习一个映射函数 f，使得相同类别的样本在特征空间中靠近，不同类别的样本被推开至少一个安全边界。",
    formula: "\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2N} \\sum_{i,j} \\left[ y_{ij} \\cdot d_{ij}^2 + (1 - y_{ij}) \\cdot \\max(0, m - d_{ij})^2 \\right]",
    contribution: "首次将“相对距离约束”引入神经网络训练，实现非参数化Metric Learning学习，在MNIST、人脸识别早期实验中验证了有效性，成为后续所有对比/三元组损失的直接灵感来源。"
  },
  {
    id: "facenet",
    title: "FaceNet: A Unified Embedding for Face Recognition and Clustering",
    authors: "Schroff et al.",
    year: 2015,
    method: "Triplet Loss",
    category: "Metric Learning",
    influences: ["contrastiveloss"],
    core_idea: "提出“三元组损失”（Triplet Loss），以其强大的判别能力成为Metric Learning学习的里程碑。其核心是：通过构建“锚点-正样本-负样本”三元组，强制锚点到正样本的距离比到负样本的距离小一个安全间隔，从而实现端到端的特征嵌入优化。",
    formula: "\\mathcal{L}_{\\text{triplet}} = \\frac{1}{N} \\sum_{i=1}^{N} \\max \\left( \\|f(x_a^i) - f(x_p^i)\\|^2 - \\|f(x_a^i) - f(x_n^i)\\|^2 + m, 0 \\right)",
    contribution: "首次在大规模人脸数据集（LFW, YCC）上实现近人类性能的识别准确率（99.63%），并证明“固定维度嵌入”可同时完成识别、聚类、验证三大任务，成为工业标准，直至今日仍在许多系统中使用。"
  },
  {
    id: "lifted",
    title: "Deep Metric Learning via Lifted Structured Feature Embedding",
    authors: "Sohn",
    year: 2016,
    method: "Lifted Structured Loss",
    category: "Metric Learning",
    influences: ["facenet"],
    core_idea: "传统三元组损失需随机采样三元组，效率低且易忽略“困难三元组”。Lifted Structured Loss（LS Loss）在每一批次内枚举所有正负样本对组合，来自动生成“最困难样本”，更高效地挖掘判别性信息。",
    formula: "\\mathcal{L}_{\\text{lifted}} = \\sum_{(i,j) \\in \\mathcal{P}} \\log \\left( \\sum_{k \\in \\mathcal{N}_i} \\exp(m - d_{ik}) + \\sum_{k \\in \\mathcal{N}_j} \\exp(m - d_{jk}) \\right)",
    contribution: "无需预先采样三元组，自动在批次内发现最困难的正负组合，显著提升收敛速度与最终性能，尤其是在类别数较少、样本分布稠密的场景中效果突出。"
  },
  {
    id: "supcon",
    title: "Supervised Contrastive Learning",
    authors: "Khosla et al.", 
    year: 2020,
    category: "Metric Learning",
    method: "SupCon",
    influences: ["simclr", "noise"],
    core_idea: "将对比学习从无监督扩展到全监督场景，突破传统对比学习仅以“样本对”为单位的限制——将同一类别的所有样本视为正样本对，不同类别样本为负样本，从而大幅提升正样本数量，显著提升类内紧凑性。",
    formula: "\\mathcal{L}_{\\text{supCon}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j \\in \\mathcal{P}_i} \\log \\frac{\\exp(s \\cdot \\cos(\\theta_{ij})) / \\tau}{\\sum_{k \\ne i} \\exp(s \\cdot \\cos(\\theta_{ik})) / \\tau}",
    contribution: "超越传统对比学习中“只能有一个正样本”的限制，通过类内全连接构建密集正样本信号，使得特征分布呈现极强的类内聚集和类间分离，在ImageNet等大规模Metric Learning任务中显著优于 Softmax，是现代SSL/半监督学习的基石之一。"
  },
  {
    id: "circleloss",
    title: "Circle Loss: A Unified Perspective of Pair Similarity Optimization",
    authors: "Sun et al.",
    year: 2020,
    method: "Circle Loss",
    category: "Metric Learning",
    influences: ["facenet", "cosface"],
    core_idea: "统一视角看待正样本对与负样本对的优化目标，提出自适应加权相似度优化机制。不同于固定间隔结构（如 Triplet），它允许根据样本对的“当前相似度”动态调整梯度强度：越像的正样本，越要推得更近；越不像的负样本，越要推得更远。",
    formula: "\\mathcal{L}_{\\text{Circle}} = \\sum_{i \\in \\mathcal{P}} \\left[ \\max(0, s_p - \\alpha_p) \\right]^2 + \\sum_{i \\in \\mathcal{N}} \\left[ \\max(0, \\alpha_n - s_n) \\right]^2",
    contribution: "首次用“对称决策边界”统一正负样本优化，实现更精细的“强度感知梯度”，显著超越 Triplet 和 Contrastive Loss，在人脸识别、行人重识别等多个任务中刷新 SOTA。"
  },
  {
    id: "rangeloss",
    title: "Range Loss for Deep Face Recognition with Long-Tailed Training Data",
    authors: "Zhang et al.",
    year: 2017,
    method: "Range Loss",
    category: "Metric Learning",
    influences: ["facenet"],
    core_idea: "针对长尾分布（少数类样本极少）导致的类间不平衡问题，提出两个独立但协同的损失项：拉近同类样本，尽可能压缩类内范围；推远异类样本，保证每类“簇”彼此分离，且类内最大距离 < 类间最小距离。",
    formula: "\\mathcal{L}_{\\text{range}} = \\max \\left( 0, \\max_{i,j \\in \\mathcal{C}_k} d_{ij} - \\min_{k \\ne l} \\min_{i \\in \\mathcal{C}_k, j \\in \\mathcal{C}_l} d_{ij} + \\lambda \\right)",
    contribution: "首次将类内最大距离与类间最小距离作为显式约束目标，适用于极端不平衡数据（如 1:1000 的身份比），在 MS-Celeb-1M 长尾子集上表现远超传统损失函数。"
  },
  {
    id: "symmetricalsiamese",
    title: "A Symmetrical Siamese Network Framework With Contrastive Learning for Pose-Robust Face Recognition",
    authors: "Zhou et al.",
    year: 2020,
    method: "Symmetrical Siamese Network",
    category: "Metric Learning",
    influences: ["contrastiveloss"],
    core_idea: "利用对称孪生结构，将同一人不同姿态的图像对作为正样本（而不同人即使正面也作为负样本），通过对比学习实现姿态不变性特征学习。",
    formula: "\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2N} \\sum_{i} \\left[ y_i \\cdot d_i^2 + (1 - y_i) \\cdot \\max(0, m - d_i)^2 \\right]",
    contribution: "证明姿态不变性无需显式建模几何变换，只需在数据处理中构造出“跨姿态正对”，即能通过对比学习自然习得不变特征，极大简化了面部姿态鲁棒性建模。"
  },
  {
    id: "deepsiamese",
    title: "Deep Siamese network for low-resolution face recognition",
    authors: "Li et al.",
    year: 2019,
    method: "Deep Siamese Network",
    category: "Metric Learning",
    influences: ["contrastiveloss"],
    core_idea: "模型训练时不依赖身份Metric Learning标签，而是以成对图像为单位构建样本：若两张图像是同一人（即使均为低清），则标签为正；若为不同人，则为负。通过对比损失优化特征空间，使低分辨率特征仍能承载身份判别信息。",
    formula: "\\mathcal{L}_{\\text{siamese}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\cdot \\log \\sigma(d_i) + (1 - y_i) \\cdot \\log (1 - \\sigma(d_i)) \\right]",
    contribution: "首次在低分辨率人脸识别中放弃超分辨率重建路径，转而直接在特征空间建立“匹配→鉴别”范式，大幅降低计算开销，适用于嵌入式设备与实时监控，在公安、交通等真实场景中被广泛借鉴。"
  },
  {
    id: "crossresolution",
    title: "Cross-resolution Learning for Face Recognition",
    authors: "Li et al.",
    year: 2019,
    method: "Cross-resolution Learning",
    category: "Metric Learning",
    influences: ["contrastiveloss"],
    core_idea: "学习一个统一的特征嵌入空间，使得同一个体的高分辨率（HR）与低分辨率（LR）图像在此空间中具有极高的相似性，从而实现跨分辨率直接比对。",
    formula: "\\mathcal{L}_{\\text{cross-res}} = \\lambda_1 \\cdot \\|f_{\\text{HR}}(x) - f_{\\text{LR}}(x)\\|_2^2 + \\lambda_2 \\cdot \\mathcal{L}_{\\text{ArcFace}}(f_{\\text{LR}}(x))",
    contribution: "提出“分辨率不变性嵌入”的显式建模方法，打破了“先超分、再识别”的传统流程，实现端到端的跨分辨率识别，在视频监控、安检系统中具有极高工程价值。"
  },
  {
    id: "coreface",
    title: "CoReFace: Sample-Guided Contrastive Regularization for Deep Face Recognition",
    authors: "Liu et al.",
    year: 2022,
    method: "CoReFace",
    category: "Metric Learning",
    influences: ["supcon"],
    core_idea: "传统对比学习中，负样本是随机采样的，容易引入“噪声负样本”（即与锚点本应相似却被误判为负）。CoReFace 提出根据当前模型对样本的置信度预测，动态筛选“有意义”的负样本——即那些模型认为“可能属于同身份”的混淆样本，用于强化训练。",
    formula: "\\mathcal{L}_{\\text{CoReFace}} = -\\log \\frac{\\exp(s \\cdot \\cos(\\theta_{ap}))}{\\exp(s \\cdot \\cos(\\theta_{ap})) + \\sum_{n \\in \\mathcal{N}_h} \\exp(s \\cdot \\cos(\\theta_{an}))}",
    contribution: "引入“模型引导的负样本选择”机制，使对比学习更聚焦于“类别边界模糊区域”，有效抑制了虚假负样本对训练的干扰，在 MS1M-v3 和 WebFace260M 等海量数据集上显著稳定训练并提升准确率。"
  },
  {
    id: "focusface",
    title: "FocusFace: Multi-task Contrastive Learning for Masked Face Recognition",
    authors: "Chen et al.",
    year: 2021,
    method: "FocusFace",
    category: "Metric Learning",
    influences: ["supcon", "bao2022beit"],
    core_idea: "在口罩遮挡下，传统方法因局部特征丢失而性能骤降。FocusFace 提出多任务对比学习框架：同时优化两个目标：全局身份对比：拉近完整脸、部分遮挡脸之间特征（正对）；局部区域聚焦：强制模型关注非遮挡区域（如眼睛、额头），通过注意力机制加权这些区域的特征贡献。",
    formula: "\\mathcal{L}_{\\text{global}} = \\mathcal{L}_{\\text{supCon}}; \\mathcal{L}_{\\text{focus}} = \\sum_{p \\in \\text{unmasked}} w_p \\cdot \\left( \\|f_p^a - f_p^p\\|^2 - \\|f_p^a - f_p^n\\|^2 \\right)",
    contribution: "首次将“遮挡鲁棒性”建模为注意力引导的对比学习任务，而非简单数据增强或特征掩码，在 RGB-D 和真实口罩数据集（WFR-Masked）上达到当时 SOTA，是疫情防控期间人脸识别技术的核心突破之一。"
  },
  
  // B. SSL对比学习 (Self-Supervised Contrastive Learning)
  {
    id: "simclr",
    title: "A Simple Framework for Contrastive Learning of Visual Representations",
    authors: "Chen et al.",
    year: 2020,
    method: "SimCLR",
    category: "SSL",
    influences: ["nce", "instadisc"],
    core_idea: "SimCLR 提出了一种无需监督标签、仅依赖数据增强的对比学习框架。其核心思想是：对同一张图像通过随机增强（如裁剪、颜色扭曲、高斯模糊等）生成两个视图，将其作为正样本对，而批次中其他所有图像的增强视图则作为负样本。模型通过对比损失函数最大化正样本对的互信息，同时最小化与负样本的相似度。",
    formula: "\\mathcal{L}_{\\text{SimCLR}} = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k=1}^{2N} \\mathbb{1}_{[k \\neq i]} \\exp(\\text{sim}(z_i, z_k) / \\tau)}",
    contribution: "系统地验证了增强策略、投影头和批次大小对性能的决定性影响，并在多个下游任务上显著超越了当时的监督与SSL方法。"
  },
  {
    id: "simclrv2",
    title: "Big Self-Supervised Models are Strong Semi-Supervised Learners",
    authors: "Chen et al.",
    year: 2020,
    method: "SimCLR v2",
    category: "SSL",
    influences: ["simclr"],
    core_idea: "SimCLR v2 在 SimCLR 基础上进行了三重升级：(1) 使用更大的 ResNet 模型；(2) 引入更深更复杂的投影头；(3) 首次系统研究了SSL预训练在半监督场景下的迁移能力。",
    formula: "\\text{沿用 InfoNCE，但通过更强的表示能力和更精细的训练策略显著提升了表示质量}",
    contribution: "为后续“大模型+无标签数据”的范式铺平了道路。"
  },
  {
    id: "moco",
    title: "Momentum Contrast for Unsupervised Visual Representation Learning",
    authors: "He et al.",
    year: 2020,
    method: "MoCo",
    category: "SSL",
    influences: ["instadisc"],
    core_idea: "MoCo 的核心创新是解决了对比学习中负样本数量受限于批次大小的问题。它引入了一个动量编码器（momentum encoder）和一个队列（queue）来存储历史样本的特征。",
    formula: "\\text{损失函数与 SimCLR 相同，但负样本来自一个动态、持久的内存库，而非瞬时批次}",
    contribution: "首次在未使用大规模批处理的情况下，让对比学习在 ImageNet 上达到与监督方法相当的性能，成为后续工作的基石。"
  },
  {
    id: "instadisc",
    title: "Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
    authors: "Wu et al.",
    year: 2018,
    method: "InstaDisc",
    category: "SSL",
    influences: ["nce"],
    core_idea: "InstaDisc 是较早提出实例级对比学习的开创性工作。它将每一个图像视作一个独立的“类别”，构建一个包含所有训练样本的记忆库（memory bank），其中存储了每个实例的特征表示。",
    formula: "\\mathcal{L}_{\\text{InstaDisc}} = -\\log \\frac{\\exp(f(x_i)^\\top m_i / \\tau)}{\\sum_{k=1}^{K} \\exp(f(x_i)^\\top m_k / \\tau)}",
    contribution: "首次在无监督条件下实现了对单一样本的精细化区分，为后续 MoCo 和 SimCLR 提供了思想原型。"
  },
  
  // 非对称网络与自蒸馏
  {
    id: "byol",
    title: "Bootstrap Your own Latent - A New Approach to Self-Supervised Learning",
    authors: "Grill et al.",
    year: 2020,
    method: "BYOL",
    category: "SSL",
    influences: ["moco"],
    core_idea: "BYOL 打破了对比学习必须依赖负样本的范式。它仅用正样本对（同一图像的两个视图），但采用非对称双网络结构：一个在线网络负责预测另一个动量更新的目标网络的表示。",
    formula: "\\mathcal{L}_{\\text{BYOL}} = \\frac{1}{2} \\| p(z_i) - z_j \\|_2^2 + \\frac{1}{2} \\| p(z_j) - z_i \\|_2^2",
    contribution: "挑战了“负样本是必须”的认知，推动了自蒸馏类方法的发展，避免了模型坍塌。"
  },
  {
    id: "simsiam",
    title: "Exploring Simple Siamese Representation Learning",
    authors: "Chen & He",
    year: 2021,
    method: "SimSiam",
    category: "SSL",
    influences: ["byol"],
    core_idea: "SimSiam 进一步简化了 BYOL，移除了动量编码器。它使用一个对称孪生网络，仅在其中一个分支上施加停止梯度（stop-gradient）操作，切断反向传播。",
    formula: "\\mathcal{L}_{\\text{SimSiam}} = -\\frac{1}{2} \\left( \\frac{p(z_1)}{\\|p(z_1)\\|_2} \\cdot \\frac{\\text{stop\\_grad}(z_2)}{\\|\\text{stop\\_grad}(z_2)\\|_2} + \\frac{p(z_2)}{\\|p(z_2)\\|_2} \\cdot \\frac{\\text{stop\\_grad}(z_1)}{\\|\\text{stop\\_grad}(z_1)\\|_2} \\right)",
    contribution: "为SSL学习提供了最轻量、最直观的实现范式，证明对抗坍塌的关键是“预测目标与自身输出分离”的结构机制。"
  },
  {
    id: "dino",
    title: "Emerging Properties in Self-Supervised Vision Transformers",
    authors: "Caron et al.",
    year: 2021,
    method: "DINO",
    category: "SSL",
    influences: ["byol", "swav"],
    core_idea: "DINO 是首个将SSL学习成功应用于 Vision Transformer（ViT）的工作，并引入了自蒸馏框架。",
    formula: "\\mathcal{L}_{\\text{DINO}} = -\\frac{1}{2} \\sum_{i} \\sum_{j} q_j^{(i)} \\log p_j^{(i)} + \\text{reversed term}",
    contribution: "其自蒸馏过程使得模型自发地学习到语义一致的聚类结构，标志着SSL学习从“判别任务”转向“生成语义结构”的拐点。"
  },
  
  // 聚类与特征空间约束
  {
    id: "invaspread",
    title: "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature",
    authors: "Ye et al.",
    year: 2019,
    method: "InvaSpread",
    category: "SSL",
    influences: ["instadisc"],
    core_idea: "在 InstaDisc 实例判别的基础上提出了双目标优化：一方面是不变性约束，另一方面新增了一个特征分散（spreading）正则项，鼓励所有实例的特征在单位超球面上均匀分布。",
    formula: "\\mathcal{L}_{\\text{InvaSpread}} = \\mathcal{L}_{\\text{invariant}} + \\lambda \\cdot \\mathcal{L}_{\\text{spreading}}",
    contribution: "首次明确分离了“正样本对拉近”与“负样本对推远”作为两类独立优化目标，并指出必须显式引入全局散度约束，深刻影响了后续方法。"
  },
  {
    id: "swav",
    title: "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments",
    authors: "Caron et al.",
    year: 2020,
    method: "SwAV",
    category: "SSL",
    influences: ["simclr"],
    core_idea: "SwAV 将对比学习从“样本-样本”层面提升至“聚类-聚类”层面。它不直接比较特征向量，而是为每个图像的两个增强视图分配软聚类标签，并强制两个视图的聚类分配应尽可能一致。",
    formula: "\\mathcal{L}_{\\text{SwAV}} = -\\frac{1}{2} \\sum_{i=1}^2 \\sum_{k=1}^K q_k^{(j)} \\log p_k^{(i)}",
    contribution: "无需显式负样本、大型队列或动量编码器，仅通过聚类对齐即可达到 SOTA 性能，为无监督表征学习提供了全新的范式。"
  },
  
  // 代理任务（Pretext Task）
  {
    id: "rotnet",
    title: "Unsupervised Representation Learning by Predicting Image Rotations",
    authors: "Gidaris et al.",
    year: 2018,
    method: "RotNet",
    category: "SSL",
    influences: [],
    core_idea: "这是早期代表性的代理任务（proxy task）类SSL方法。给定一张图像，将其随机旋转 0°、90°、180° 或 270°，然后让模型预测其旋转角度。",
    formula: "\\mathcal{L}_{\\text{Rot}} = -\\sum_{c=1}^{4} y_c \\log p_c",
    contribution: "证明了通过设计一个合理、易于优化的SSL代理任务，同样可以提取出具有判别力的特征，为后续“SSL即代理任务”的范式铺平了道路。"
  },
  
  // 理论基石与跨领域启发
  {
    id: "nce",
    title: "Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics",
    authors: "Gutmann & Hyvärinen",
    year: 2010,
    method: "NCE",
    category: "SSL",
    influences: [],
    core_idea: "NCE 作为现代对比学习的理论基石，最早用于从无归一化（unnormalized）的概率模型中估计参数。其基本思想是将“概率密度估计”转化为一个二元Metric Learning问题。",
    formula: "\\mathcal{L}_{\\text{NCE}} = -\\mathbb{E}_{x \\sim p_d} \\left[ \\log \\frac{p_d(x)}{p_d(x) + k p_n(x)} \\right] - \\mathbb{E}_{x \\sim p_n} \\left[ \\log \\frac{k p_n(x)}{p_d(x) + k p_n(x)} \\right]",
    contribution: "将SSL学习从“启发式技巧”提升为“可证明的统计推断框架”，为 InfoNCE、SimCLR、MoCo 的损失函数提供了理论解释。"
  },
  {
    id: "word2vec",
    title: "Distributed Representations of Words and Phrases and their Compositionality",
    authors: "Mikolov et al.",
    year: 2013,
    method: "Word2Vec",
    category: "SSL",
    influences: [],
    core_idea: "虽然 Word2Vec 并非视觉方法，但其提出的“上下文预测”机制是现代SSL学习的鼻祖。它通过局部共现的统计模式在向量空间中编码语义关系。",
    formula: "\\mathcal{L}_{\\text{Skip-gram}} = \\sum_{(w, c) \\in D} \\log \\sigma(v_c^\\top v_w)",
    contribution: "揭示了“关系即结构”的思想，被直接迁移到对比学习中，是从“统计学习”迈向“几何表征学习”的关键一步。"
  },
  {
    id: "simcse",
    title: "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    authors: "Gao et al.",
    year: 2021,
    method: "SimCSE",
    category: "SSL",
    influences: ["simclr"],
    core_idea: "SimCSE 将对比学习思想成功迁移到自然语言处理中的句向量生成任务。它仅依赖标准 Dropout 作为无损数据增强。",
    formula: "\\mathcal{L}_{\\text{SimCSE}} = -\\log \\frac{\\exp(\\text{sim}(z_i, z_i^+) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(z_i, z_j) / \\tau)}",
    contribution: "揭示了模型内部的随机性本身即可作为强大的增强信号，成为后续句嵌入的基准。"
  },
  
  // C. 图对比学习 (Graph Contrastive Learning)
  {
    id: "graphcl",
    title: "Graph Contrastive Learning with Augmentations",
    authors: "You et al.",
    year: 2020,
    method: "Graph Contrastive Learning",
    category: "SSL",
    influences: ["simclr"],
    core_idea: "将对比学习扩展到图结构数据，通过对图结构或节点特征进行随机扰动（如边丢弃、特征遮罩），生成两个“视图”，然后最大化同一个节点在两个视图中的表示互信息。",
    formula: "\\mathcal{L}_{\\text{GCL}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(\\text{sim}(z_i^{(1)}, z_i^{(2)}) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(z_i^{(1)}, z_j^{(2)}) / \\tau)}",
    contribution: "在人脸识别中，该思想被用于构建“身份关系图”：节点为人脸，边为身份相似性，通过图增强模拟姿态、光照变化，实现无监督身份特征学习，在标注数据稀缺时效果显著。"
  },
  {
    id: "con2r",
    title: "Graph-Constrained Contrastive Regularization for Semi-weakly Volumetric Segmentation",
    authors: "Zhang et al.",
    year: 2022,
    method: "Graph-Constrained Contrastive Regularization",
    category: "SSL",
    influences: ["graphcl"],
    core_idea: "虽非专为人脸识别设计，但其“图约束对比正则化”思想极具迁移价值。其核心是：在对比学习中引入图拓扑结构作为软约束，确保相似节点在特征空间中不仅相似，还需满足图连接关系（如邻居应更近）。",
    formula: "\\mathcal{L}_{\\text{graph-reg}} = \\sum_{(i,j) \\in E} w_{ij} \\cdot \\|z_i - z_j\\|^2",
    contribution: "将“先验结构知识”注入对比学习，避免模型学习到虚假的聚类关系，在弱监督条件下实现更稳定、语义一致的特征分布，后续已被广泛应用于跨模态对齐与联合表示学习。"
  },
  {
    id: "simgcl",
    title: "Are Graph Augmentations Necessary?: Simple Graph Contrastive Learning for Recommendation",
    authors: "Wan et al.",
    year: 2022,
    method: "SimGCL",
    category: "SSL",
    influences: ["graphcl"],
    core_idea: "颠覆传统观点，提出“无需复杂图增强”的简单对比学习范式。SimGCL 仅通过对节点嵌入进行轻微高斯噪声扰动，即可生成两个有效视图，性能却优于复杂图采样增强方法。",
    formula: "\\mathcal{L}_{\\text{SimGCL}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(\\text{sim}(z_i', z_i'') / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(z_i', z_j'') / \\tau)}",
    contribution: "证明在特征空间内的微扰比结构扰动更有效、更稳定，为人脸识别中的“轻量级成员扰动对比”提供了新思路。"
  },
  
  // 第三部分：分析与综述 (Analysis & Surveys)
  {
    id: "significance",
    title: "Significance of Softmax-Based Features in Comparison to Distance Metric Learning-Based Features",
    authors: "Horiguchi et al.",
    year: 2020,
    method: "Analysis",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "这篇工作系统比较了两类特征学习范式：一类以 Softmax Metric Learning（如 ImageNet 监督Metric Learning）为损失函数，另一类以Metric Learning学习（如 Triplet Loss、Contrastive Loss）为优化目标。",
    formula: "N/A",
    contribution: "这一结论颠覆了当时“Metric Learning学习更优”的主流预期，并推动了后续基于 Softmax 的改进方法成为人脸识别的标准方案。它提醒我们：监督信号的强度和损失函数的结构设计同样重要，不能单纯依赖“对比”的形式化机制。"
  },
  {
    id: "noise",
    title: "The Devil of Face Recognition Is in the Noise",
    authors: "Wang et al.",
    year: 2018,
    method: "Data Analysis",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "这是一篇极具现实意义的数据分析论文，作者分析了超过 10 个主流人脸识别数据集（包括 MS-Celeb-1M、WebFace、LFW），发现其中存在大量标签噪声。",
    formula: "N/A",
    contribution: "彻底改变了研究社区对数据质量的认知，强调“数据比模型更重要”。其结论推动了多个数据清洗工具、噪声鲁棒训练策略的发展，是SSL学习从“理想实验”走向“真实世界”的重要警钟。"
  },
  
  // 新增的论文
  {
    id: "otclip",
    title: "OT-CLIP: Understanding and Generalizing CLIP via Optimal Transport",
    authors: "Long et al.",
    year: 2022,
    method: "Optimal Transport",
    category: "SSL",
    influences: ["radford2021clip"],
    core_idea: "指出标准 CLIP 模型通过全局特征进行图文匹配，导致对齐粒度过粗，难以理解局部细节。OT-CLIP 提出将图文匹配问题重新建模为一个最优传输 (Optimal Transport, OT) 问题，旨在寻找图像块 (Patch) 与文本次元 (Word Token) 之间最经济、最合理的细粒度对应关系。",
    formula: "\\mathcal{L}_{\\text{OT}}(C) = \\min_{P \\in \\Pi(r, c)} \\langle P, C \\rangle_F",
    contribution: "提供了细粒度对齐框架，增强了解释性，提升了泛化性能。"
  },
  {
    id: "cr2pq",
    title: "CR2PQ: Continuous Relative Rotary Positional Query for Dense Visual Representation Learning",
    authors: "Zhang et al.",
    year: 2023,
    method: "Continuous Relative Rotary Positional Query",
    category: "SSL",
    influences: ["rope"],
    core_idea: "针对密集视觉任务（如分割、检测），传统的位置编码在随机裁剪等数据增强下会失效。本文提出一种新颖的连续相对旋转位置查询机制，让模型学习一种与绝对位置无关、仅依赖于相对空间关系的位置感知能力。",
    formula: "f(x, m) = x \\cdot e^{im\\theta}",
    contribution: "提出了一种全新的相对位置编码范式，完美适配密集预测任务，统一了内容与位置学习。"
  },
  {
    id: "basealign",
    title: "Align Representations with Base: A New Approach to Self-Supervised Learning",
    authors: "Zhang et al.",
    year: 2022,
    method: "BaseAlign",
    category: "SSL",
    influences: ["byol", "simsiam"],
    core_idea: "挑战了主流SSL学习范式。它认为，无论是需要大量负样本的对比学习，还是需要动量编码器和预测头的非对比学习，都过于复杂。BaseAlign 提出一种极简的非对比学习框架：只需将当前模型对图像的表示，与该图像在先前训练阶段的“旧”表示进行对齐即可。",
    formula: "\\mathcal{L} = \\left\\| \\frac{z_i}{\\|z_i\\|_2} - \\frac{b_i}{\\|b_i\\|_2} \\right\\|_2^2",
    contribution: "提供了迄今最简单的非对比学习框架之一，揭示了避免模型坍塌的新机制，实现了高效的实现。"
  },
  {
    id: "pq",
    title: "Patch-Level Contrastive Learning via Positional Query for Visual Pretraining",
    authors: "Zhang et al.",
    year: 2023,
    method: "Positional Query",
    category: "SSL",
    influences: ["simclr", "moco"],
    core_idea: "为了解决在随机裁剪下进行 Patch 级别对比学习时“Patch 对应关系丢失”的难题，本文提出一种基于位置查询的新范式。它不直接匹配 Patch 特征，而是创建一组可学习的、代表抽象空间概念的“位置查询”向量。",
    formula: "\\mathcal{L} = -\\sum_{k=1}^{K} \\left( \\text{sim}(A_{1,k}, A_{2,k}) - \\log \\sum_{j=1}^{K} \\exp(\\text{sim}(A_{1,k}, A_{2,j})) \\right)",
    contribution: "优雅地解决了 Patch 对应难题，学习了内容与空间的解耦表示，在密集预测任务上表现出色。"
  },
  {
    id: "iotcl",
    title: "Understanding and Generalizing Contrastive Learning from the Inverse Optimal Transport Perspective",
    authors: "Various",
    year: 2022,
    method: "Inverse Optimal Transport",
    category: "SSL",
    influences: ["nce"],
    core_idea: "首次将对比学习的优化目标与逆最优传输联系起来，提供了一个全新的理论框架。传统对比学习通过拉近正样本、推远负样本来学习表示，但其数学基础长期缺乏统一解释。",
    formula: "\\mathcal{L}_{\\text{CL}} = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j)/\\tau)}{\\sum_{k \\neq i} \\exp(\\text{sim}(z_i, z_k)/\\tau)}",
    contribution: "首次建立对比学习与OT的深层数学联系，为设计更鲁棒的负样本采样策略提供理论指导。"
  },
  {
    id: "dinov3",
    title: "DINOv3",
    authors: "Various",
    year: 2023,
    method: "DINOv3",
    category: "SSL",
    influences: ["dino", "dinov2"],
    core_idea: "DINOv3 是 DINOv2 的重大升级，目标是构建一个通用型视觉基础模型，无需微调即可在图像、像素、多尺度任务中取得超越专用模型的性能。",
    formula: "\\mathcal{L}_{\\text{Gram}} = \\| G_{\\text{student}} - G_{\\text{teacher}} \\|_F^2",
    contribution: "DINOv3 在 10+ 个视觉任务上全面超越 CLIP、SAM 等模型，且无需微调，是真正“开箱即用”的通用特征提取器。"
  },
  {
    id: "dinov2",
    title: "DINOv2: Learning Robust Visual Features without Supervision",
    authors: "Caron et al.",
    year: 2023,
    method: "DINOv2",
    category: "SSL",
    influences: ["dino"],
    core_idea: "DINOv2 的目标是证明：仅靠无监督学习，即可训练出媲美有监督预训练的通用视觉特征表示，为计算机视觉提供“类 BERT”的基础模型。",
    formula: "\\mathcal{L}_{\\text{DINO}} = \\sum_{t} \\text{KL} \\left( \\text{softmax}(z_t / \\tau) \\| \\text{softmax}(\\bar{z}_t / \\tau) \\right)",
    contribution: "DINOv2 的特征在 ImageNet 零样本线性评估中达到 80.1%，首次超越 OpenCLIP。"
  },
  {
    id: "maskfeat",
    title: "Masked Feature Prediction for Self-Supervised Visual Pre-Training",
    authors: "Various",
    year: 2021,
    method: "MaskFeat",
    category: "SSL",
    influences: ["mae"],
    core_idea: "MaskFeat 是一种基于特征预测的SSL学习方法，不是重建像素，而是重建人类设计的特征（如 HOG）。这种方法更高效，在视频理解中实现突破。",
    formula: "\\mathcal{L}_{\\text{MaskFeat}} = \\sum_{m \\in M} \\| \\hat{f}_m - f_m \\|_2^2",
    contribution: "在 Kinetics-400 上以 MViT-L 达到 86.7%，远超同期方法，且计算效率极高。"
  },
  {
    id: "mae",
    title: "Masked Autoencoders Are Scalable Vision Learners",
    authors: "He et al.",
    year: 2021,
    method: "MAE",
    category: "SSL",
    influences: ["devlin2019bert"],
    core_idea: "MAE 是视觉领域 BERT 的图灵级实现，通过遮蔽公式与不对称编解码结构，实现了高效且强大可扩展的SSL视觉预训练。",
    formula: "\\mathcal{L}_{\\text{MAE}} = \\frac{1}{N} \\sum_{i=1}^N \\| \\hat{x}_i - x_i \\|_2^2",
    contribution: "能训练 ViT-Huge，在 ImageNet-1K 上达到 87.8% 线性评估准确率，超越监督预训练。"
  }
];

// Formulas have been converted to valid LaTeX format
const latent = [
  // =================================================================================
  // 第一部分：Core Models & Architectures与显式推理 (Foundational & Explicit Reasoning)
  // =================================================================================
  {
    id: "wei2022cot",
    title: "Chain-of-thought prompting elicits reasoning in large language models",
    authors: "Wei et al.",
    year: 2022,
    method: "Chain-of-Thought (CoT) Prompting",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "通过在提示中提供详细的推理步骤范例，引导大型语言模型在回答问题前先生成一个连贯的、中间的“思维链”，从而解锁其内在的复杂推理能力。",
    formula: "Q \\rightarrow A \\quad \\Rightarrow \\quad Q \\rightarrow \\text{CoT} \\rightarrow A",
    contribution: `这篇论文是大型语言模型推理领域的里程碑式作品，它正式提出并验证了“思维链”（Chain-of-Thought, CoT）提示方法的有效性。在此之前，提升大型语言模型（LLM）在复杂推理任务（如算术、常识和符号推理）上表现的主要方法是单纯地扩大模型规模。然而，即使是极大规模的模型，在需要多步逻辑推导的问题上仍然表现不佳。这篇论文的作者发现，这种失败并非源于模型能力的缺失，而是在于未能引导模型展现其潜在的推理能力。

核心思想与方法：论文的核心思想极为简洁而深刻：与其让模型直接输出最终答案，不如引导它“像人一样思考”，即先把解决问题的中间步骤一步步地写出来，最后再给出结论。这种方法被称作思维链提示。具体实现上，研究者在给模型的少量样本（few-shot）提示中，不仅提供了“问题-答案”对，还加入了详细的“问题-推理过程-答案”的范例。当模型学习到这种格式后，在面对新问题时，它也会模仿这种模式，自主地生成一系列连贯的、解释性的中间推理步骤，从而极大地提高了最终答案的准确性。

主要贡献与发现：
1.  **引发推理能力**：论文证明了 CoT 能够成功“解锁”或“引出”LLM 内在的、但未被充分利用的推理能力。对于之前模型难以解决的任务，CoT 带来了惊人的性能提升。
2.  **规模效应的涌现**：一个非常重要的发现是，CoT 的效果具有显著的“规模依赖性”。对于较小的模型，CoT 的效果并不明显甚至可能有害。但当模型规模超过某个阈值后，CoT 带来的性能增益会突然涌现并急剧增长，这表明多步推理能力是大型模型的一种“涌现能力”。
3.  **可解释性与可除错性**：CoT 生成的推理链为理解模型的“思考过程”提供了一个窗口。当模型出错时，研究者可以检查其中间步骤，定位到具体的错误环节。

这篇论文为《A Survey on Latent Reasoning》提供了重要的背景。它所确立的“显式思维链”是所有“潜在推理”方法试图改进或超越的对象，旨在解决 CoT 依赖自然语言、表达带宽受限等问题。`
  },

  // =================================================================================
  // 第二部分：Latent Reasoning Paradigms (Vertical Recurrence & Dynamic Computation)
  // =================================================================================
  {
    id: "dehghani2018ut",
    title: "Universal transformers",
    authors: "Dehghani et al.",
    year: 2018,
    method: "Universal Transformer (UT) with ACT",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "将固定的 Transformer 层级结构转变为一个时间上的循环过程，并引入自适应计算时间（ACT）机制，使模型能根据输入复杂度动态决定每个位置的计算深度。",
    formula: "p_t = \\sigma(W_h h_t^l + b_h)",
    contribution: `这篇论文是探索动态计算和循环机制在 Transformer 架构中应用的开创性工作，为后来的“垂直循环”潜-在推理方法奠定了基础。标准 Transformer 模型由固定数量的层组成，计算深度是静态的，缺乏灵活性。

核心思想与方法：论文提出了“通用 Transformer”（Universal Transformer, UT），其核心思想是将 Transformer 的层级结构转变为一个时间上的循环过程。UT 模型只有一个（或一小组）可循环使用的核心层。对于输入的每一个 token，UT 会对其表示进行多轮迭代式的更新。为了让模型能动态决定计算轮数，UT 引入了“自适应计算时间”（Adaptive Computation Time, ACT）机制。ACT 是一个可学习的模组，它在每一轮计算后会输出一个“停止概率”。当某个位置的累计停止概率达到阈值时，该位置的计算就会停止。这样一来，模型可以为序列中的不同部分分配不同的计算资源。

主要贡献与发现：
1.  **计算通用性**：论文从理论上证明了通用 Transformer 是图灵完备的，具备执行任何可计算函数的潜力，而标准 Transformer 则不具备。
2.  **动态计算深度**：UT 成功地将网络深度从一个固定的超参数变为一个动态的、依赖于输入的变量。ACT 机制使得计算资源的分配更加智慧和高效。
3.  **潜-在推理的架构基础**：UT 的循环计算机制是“垂直循环”潜-在推理的直接体现。它让模型在不生成任何中间文本的情况下，在内部隐藏状态中进行多步“思考”和“精炼”。《A Survey on Latent Reasoning》将其视为通过显式架构设计实现潜-在推理的典范，后续的 CoTFormer、Recurrent-Depth 等工作都是在其思想上进行的扩展。`
  },
  {
    id: "mohtashami2023cotformer",
    title: "Cotformer: A chain-of-thought driven architecture with budget-adaptive computation cost at inference",
    authors: "Mohtashami et al.",
    year: 2023,
    method: "CoTFormer with Mind-of-Routers (MoR)",
    category: "Latent Reasoning Paradigms",
    influences: ["dehghani2018ut", "wei2022cot"],
    core_idea: "受 CoT 启发，设计了一种预算自适应的动态计算架构，通过一个“思想路由器”在每个步骤决定是对当前 token 进行更深层次的循环精炼，还是将其传递到下一阶段。",
    formula: "\\text{action} = \\text{MoR}(h_t^l) \\in \\{\\text{PONDER}, \\text{PASS}\\}",
    contribution: `这篇论文直面了标准 Transformer 模型计算成本固定的问题，提出了一种新颖的、受思维链（CoT）启发的动态计算架构——CoTFormer。其核心目标是让模型学会在给定的计算预算内，智慧地分配计算资源，对问题的不同部分进行不同深度的处理。

核心思想与方法：CoTFormer 的架构精髓在于其“预算自适应计算”。它将 Transformer 的层块组织成一个可以重复执行的共享计算单元。模型在处理输入序列时，通过一个名为“思想路由器”（Mind-of-Routers, MoR）的模组来决定下一步的计算策略。这个 MoR 是一个可学习的门控网络，它会评-估当前 token 的隐藏状态，并决定是应该“深入思考”（PONDER，即将该 token 的表示送回共享层块进行新一轮的精炼），还是“通过”（PASS，将其传递到下一阶段）。这种机制使得计算深度变成了 token 级-别的动态变量。对于简单的 token，MoR 可能很快就决定停止计算；而对于需要复杂推理的关键 token，模型则可以反覆将其送入共享层进行多达数十次的迭代处理。

主要贡献与发现：
1.  **动态与自适应计算**：CoTFormer 成功地将 Transformer 从一个静态计算图转变为一个动态的、内容驱动的计算过程，为“垂直循环”潜-在推理提供了一个非常具体的、可操作的架构实现。
2.  **效率与性能的平衡**：实验证明，CoTFormer 能够在使用更少平均计算成本的同时，保持甚至超越标准 Transformer 在复杂推理任务上的性能。
3.  **与潜-在推理的联系**：模型对 token 进行的多轮迭代精炼，本质上就是在不生成任何文本的情况下，在潜-在空间中进行多步推理。每一次循环都可以被看作是潜-在思维链中的一步。`
  },
  {
    id: "bae2024relaxed",
    title: "Relaxed recursive transformers: Effective parameter sharing with layer-wise lora",
    authors: "Bae et al.",
    year: 2024,
    method: "Relaxed Recursive Transformers",
    category: "Latent Reasoning Paradigms",
    influences: ["dehghani2018ut"],
    core_idea: "为解决循环 Transformer 中“完全权重共享”导致的表征瓶颈问题，提出一种半共享权重机制：主体参数共享，但为每个循环深度配备独立的、小规模的 LoRA 适配器，赋予不同计算深度专门化的功能。",
    formula: "W_{\\text{eff}}^{(d)} = W_{\\text{shared}} + W_{\\text{LoRA}}^{(d)}",
    contribution: `这篇论文关注的是循环式 Transformer 架构在实际训练和应用中的一个核心挑战：参数共享的僵化性。像 Universal Transformer 这样的早期模型在所有循环步骤中共享完全相同的权重，这虽然节省了参数，但也导致模型很难在不同的计算深度上学习到功能各异的、专门化的表征。

核心思想与方法：论文提出了“松弛递归 Transformer”。其核心创新在于引入了一种“半共享”的权重机制。模型的主体参数在所有递归步骤中是共享的，但它为每一个递归“深度”都配备了一组独立的、小规模的 LoRA（Low-Rank Adaptation）模组。这意味着模型在第 d 次循环时，其有效的权重是共享权重与深度 d 专属的 LoRA 权重的和。由于 LoRA 模组的参数非常少，这种设计既保留了参数高效共享的主要优-点，又赋予了模型在不同计算深度上进行功能分化的能力，例如在浅层递归中提取基本特征，在深层递归中执行抽象的逻辑操作。

主要贡献与发现：
1.  **灵活的参数共享**：该工作为循环式架构的参数共享问题提供了一个优-雅且高效的解决方案，证明了通过 LoRA 进行层级化的、深度的特定微调是可行且有效的。
2.  **提升算法推理能力**：实验表明，“松弛递归 Transformer”的性能显著优-于传统固定深度 Transformer 和严格共享参数的 Universal Transformer，证明了赋予不同计算深度特定功能的重要性。
3.  **对垂直循环的深化**：这项研究深化了对垂直循环潜-在推理机制的理解。它揭示了有效的潜-在推理不仅需要足-够的计算深度，还需要在这些深度上实现功能上的“分工”，为设计更强大的潜-在推理模型提供了新的思路。`
  },
  {
    id: "geiping2025recurrentdepth",
    title: "Scaling up test-time compute with latent reasoning: A recurrent depth approach",
    authors: "Geiping et al.",
    year: 2025,
    method: "Recurrent-Depth",
    category: "Latent Reasoning Paradigms",
    influences: ["dehghani2018ut", "bae2024relaxed"],
    core_idea: "将 Transformer 架构清晰地划分为“前奏-循环-尾声”三阶段，将迭代推理过程模组化。模型可在测试时通过增加核心循环部分的迭代次数，灵活地用计算换取更高性能，以解决更复杂的问题。",
    formula: "x_{t+1} = \\text{LoopBlock}(x_t)",
    contribution: `这篇论文提出了一种名为“循环深度”（Recurrent-Depth）的架构，旨在系统性地提升 Transformer 模型在测试时的计算能力。该工作基于一个核心观察：许多复杂推理任务可以被形式化为求解一个不动点方程（fixed-point equation）。

核心思想与方法：Recurrent-Depth 模型将 Transformer 架构划分为三个阶段：
*   **前奏 (Prelude)**：一个标准 Transformer 编码器，负责对输入进行初步编码。
*   **循环 (Loop)**：模型的核心。一个共享的 Transformer 层，对前奏部分的输出进行多轮迭代式的精炼。
*   **尾声 (Coda)**：一个标准 Transformer 解码器，接收循环部分最终的精炼表示，并生成答案。

这种结构化的设计使得计算的核心——迭代推理——被清晰地模组化。作者探索了不同的循环停止机制，从固定迭代次-数到基于不动点收敛的动态停止准则。这种设计使得模型在测试时，可以简单地通过增加循环次-数来换取更强的推理性能，实现计算成本和准确率之间的灵活权衡。

主要贡献与发现：
1.  **结构化的垂直循环**：论文提出的“前奏-循环-尾声”结构，为设计垂直循环的潜-在推理模型提供了一个清晰、模组化且可扩展的蓝图。
2.  **与不动点求解的联系**：该工作将深度神-经网络中的迭代计算与数学中的不动点求解深刻地联系起来，为理解潜-在推理的计算本质提供了一个理论视角。
3.  **计算的可扩展性**：实验证明，在测试时简单地增加迭代次-数，就能让模型解决训练时从未见过的、更复杂的问题实例，充分展示了潜-在推理在“泛化到更难问题”方面的巨大潜力。`
  },
  {
    id: "zeng2025ponder",
    title: "Pretraining language models to ponder in continuous space",
    authors: "Zeng et al.",
    year: 2025,
    method: "Pondering Cycle Pre-training",
    category: "Latent Reasoning Paradigms",
    influences: ["dehghani2018ut"],
    core_idea: "提出一种在预训练阶段就引入“思考”机制的方法。在预测每个词前，模型会执行一个内部的“思考循环”：先生成一个初步的概率分布，将其转为连续的“思考嵌入”向量，再反馈回隐藏状态进行多轮迭代精炼，让“深思熟虑”成为模型的内-在能力。",
    formula: "h_{t+1} = h_t + \\text{Linear}(\\sum_{v \\in V} P(v|h_t) \\cdot E_v)",
    contribution: `这篇论文提出了一-种极具创新性的架构和预训练方法，旨在让语言模型从训练之初就学会“深思熟虑”。传统模型在预测下-一个词时通常是“一锤子买卖”，而这-项工作认为，模型应该被赋予在做出最终决策前，对其内-部的“想法”进行多轮迭代精炼的能力。

核心思想与方法：论文的核心是引入了一个“思考循环”（pondering cycle）机制，并将其整合进预训练的每一个步骤中。对于每一个需要预测的 token，其计算过程不再是单次前向传播，而是：
1.  **初步预测**：模型首先计算出词汇表的概率分布。
2.  **形成“思考嵌入”**：接着，模型将此分布转-化为一个连续的“思考嵌入”，即整个词汇表嵌入的加权平均。
3.  **反馈与精炼**：这个“思考嵌入”会通过残差连接，被重新加回到当-前的隐藏状态中。
4.  **迭代循环**：模型会带着这个新的隐藏状态，重新执行同一个 Transformer 层的计算，重复上述步骤 k 次。

通过在预训练的每一步都强制模型进行 k 次这样的“思考循环”，模型被迫学习到一种能够支-持迭代式精炼的表征，其参数从一开始就被优-化来适应这种“反覆琢磨”的计算流。

主要贡献与发现：
1.  **将思考融入预训练**：这-项工作首次将潜-在推理的迭代机制，从微调阶段推向了通用语言模型的预训练核心，旨在让“思考”成为模型的一种基础能力。
2.  **架构上的潜-在推理**：它提出了一-种新颖的、基于激活值反-馈的“垂直循环”架构，是一-种全新的、细粒度的潜-在推理形式。
3.  **提升困难任务性能**：实验表明，经过这种方式预训练的模型，在需要深度思考的复杂推理任务上，相较于传统预训练模型表现更佳，为构-建更强大的基础模型提供了全新的架构设计思路。`
  },

  // =================================================================================
  // 第三部分：Training & Optimization (Training-Induced & Compression)
  // =================================================================================
  {
    id: "hao2024coconut",
    title: "Training large language models to reason in a continuous latent space (Coconut)",
    authors: "Hao et al.",
    year: 2024,
    method: "Coconut (Continuous CoT)",
    category: "Training & Optimization",
    influences: ["wei2022cot", "deng2023internalize"],
    core_idea: "通过课程学习和强化学习，逐步用一个可学习的“连续思考向量”替代显式 CoT 中的中间文本步骤，最终训练出一个无需生成任何 token、完全在潜-在空间中进行多步推理的模型，大幅提升推理速度。",
    formula: "\\pi_{\\theta}(\\text{action} | \\text{state}) \\rightarrow \\text{action} = \\text{thought\\_vector}",
    contribution: `这篇论文是“训练诱导循环”潜-在推理方向的代表性工作。它旨在解决显式 CoT 的一个核心痛点：生成中间推理步骤会消耗大量的时-间和计算资源，并且推理过程受限于离散的自然语言。

核心思想与方法：这篇论文的核心方法名-为“Coconut”（Continuous Chain-of-Thought），其目标是训练一个模型，使其能够-在不生成文本的情况下，执行类似 CoT 的多步推理。它通过一个巧妙的课程学习和策略梯度方法来实现。训练过程分为两个阶段：
1.  **模仿学习与逐步压缩**：首先，模型在包含显式 CoT 的数据-集上训练。接着，进入一个课程学习阶段，逐步、随机地用一个特殊的可学习向量（“连续思考向量”）替换掉 CoT 中的部分中间步骤，迫使模型学会将越来越长的推理逻辑“压缩”进这个连续的思考向量中。
2.  **强化学习微调**：当模型能够-完全用思考向量替代所有中间步骤后，转而使用强化学习进行微调。模型在每一步生成一个思考向量，形成一条“潜-在的”思维链。只有在最后一步，模型才需要生成最终答案。如果答案正确，模型会获得奖励。

主要贡献与发现：
1.  **实现全潜-在推理**：Coconut 成功地训练出了一-个能够在连续潜-在空间中进行多步推理的模型。在推理时，它不需要生成任何中间 token，从而大大提高了推理速度。
2.  **广度优先搜索**：由于思考向量是连续的，模型可以在潜-在空间中同时探索多条不同的推理路径，这类似一种并行的、广度优先的搜索，理论上能找到更优-的解决方案。
3.  **训练诱导循环的范例**：这篇论文证明了即使在标准的 Transformer 架构上，也可以通过精心设计的训练策略来“诱导”出循环计算的行为，而无需对模型架构做任何修改。`
  },
  {
    id: "deng2023internalize",
    title: "From explicit cot to implicit cot: Learning to internalize cot step by step",
    authors: "Deng et al.",
    year: 2023,
    method: "Stepwise Internalization",
    category: "Training & Optimization",
    influences: ["wei2022cot"],
    core_idea: "通过知识蒸馏，让一个“学生”模型（不生成 CoT）的内部隐藏状态，去对齐一个“教师”模型（生成完整 CoT）在完成推理后的“思维总结”状态，从而将序列化的推理过程“编译”进一次前向传播中。",
    formula: "\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\lambda \\cdot \\| h_{\\text{student}}^{\\text{final}} - h_{\\text{teacher}}^{\\text{CoT-final}} \\|_2^2",
    contribution: `这篇论文提出了一种名为“逐步内化”的精巧训练方法，旨在将 LLM 从依赖外部、冗长的显式思维链，转变为能够在内部高效执行隐式思维链的推理器。

核心思想与方法：该方法采用了知识蒸馏和课程学习的策略。首先，一个“教师模型”被训练来生成完整的、带详细中间步骤的显式 CoT。然后，一个“学生模型”的任务是直接输出最终答案。训练的关键在于一个特殊的“内化损失”。研究者会提取教师模型在生成最终答案前、位于推理链末端的那个关键隐藏状态，这个状态被认为是教师模型“想清楚了”之后的总结。训练目标就是让学生模型在进行单步前向传播时，其内部某一层（或多层）的隐藏状态，与教师模型那个关键的“总结状态”尽可能地相似。通过这种方式，学生模型被迫学会在其自身的层级结构中，模拟并压缩教师模型多步、序列化的推理过程。

主要贡献与发现：
1.  **高效的推理内化**：论文证明了可以将一个序列化的、外显的推理过程“编译”进一个并行的、内隐的神-经网络计算过程中，使得模型在推理时速度极快。
2.  **训练诱导循环的典范**：这项工作是“训练诱导循环”的绝佳范例，清晰地展示了，即便不修改 Transformer 架构，也能通过巧妙的训练目标，诱导模型在层与层之间形成一个隐式的、类似 CoT 的计算流。
3.  **加深对层级功能的理解**：该方法从实践上证明了 Transformer 的层级堆叠可以被塑造成一个有结构的推理管道，每一层都执行着被压缩的、更深一步的推理。`
  },
  {
    id: "shen2025codi",
    title: "Codi: Compressing chain-of-thought into continuous space via self-distillation",
    authors: "Shen et al.",
    year: 2025,
    method: "Codi (Compressing via self-distillation)",
    category: "Training & Optimization",
    influences: ["hao2024coconut", "deng2023internalize", "wei2022cot"],
    core_idea: "提出一种简洁高效的“单步自蒸馏”框架，使用模型自身作为教师，让不生成 CoT 的“学生”路径的最终隐藏状态，去对齐生成了完整 CoT 的“教师”路径的最终隐藏状态，首次在数学推理任务上实现了与显式 CoT 性能持平。",
    formula: "\\mathcal{L}_{\\text{align}} = \\| h_{\\text{student}}^{\\text{final}} - \\text{stop\\_grad}(h_{\\text{teacher}}^{\\text{final}}) \\|_2^2",
    contribution: `这篇论文在“训练诱导循环”方向上取得了一项重大突破，提出了一种名为 Codi 的高效方法，用于将显式的 CoT 推理过程，无损地压缩到一个连续的潜-在空间中。

核心思想与方法：Codi 的精髓在于其“单步自蒸馏”框架。它使用模型自身作为教师，来教导自己如何进行隐式推理。
*   **教师模型**：被允许生成完整的、显式的 CoT 推理步骤。
*   **学生模型**：被严格禁止生成任何中间 CoT 步骤，必须直接输出最终答案。
训练的关键在于一个“思维对齐损失”。研究者提取教师模型在完成了所有 CoT 推理、即将生成最终答案时的那个隐藏状态向量，这个向量被认为是教师模型对整个问题“深思熟虑”后的最终“思维结晶”。Codi 的核心目标，就是让学生模型在仅进行一次前向传播后，其最后一层-的隐藏状态，与教师模型的这个“思维结晶”在向量空间中尽可能地接近。

主要贡献与发现：
1.  **推理压缩的里程碑**：Codi 是首个在主流数学推理基准（GSM8K）上，实现了与显式 CoT 推理性能持平的潜-在推理方法，雄辩地证明了复杂推理链完全可以被压缩进一个静态的向量表示中。
2.  **简洁高效的训练**：相比之前复杂的训练流程，Codi 的单步自蒸馏方法更加稳定、高效，更容易复现，为将现有强大 LLM 的推理能力“内化”提供了一-条标准化的技术路径。
3.  **潜-在推理的实证基础**：这项工作为潜-在推理的有效性提供了最强有力的实证支持之一，证明了模型的潜-在空间具有足-够的容量和表达能力来容纳复杂的逻辑推导。`
  },

  // =================================================================================
  // 第四部分：Latent Reasoning Paradigms (Horizontal Recurrence & Long Context)
  // =================================================================================
  {
    id: "peng2024rwkv",
    title: "Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence",
    authors: "Peng et al.",
    year: 2024,
    method: "RWKV with Matrix-Valued States",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "将 RWKV 模型的循环状态从向量升级为矩阵，以更结构化地存储历史信息。同时引入数据依赖的动态递归机制，让模型能根据当前输入智慧地调整记忆的衰减率，在保持 O(1) 推理效率的同时大幅增强长程依赖建模能力。",
    formula: "S_t = f(S_{t-1}, x_t, g(x_t))",
    contribution: `这篇论文是 RWKV（Receptance Weighted Key Value）架构的一个重要演进。RWKV 本身就是结合了 Transformer 的性能和 RNN 的高效推理的代表性模型。

核心思想与方法：论文提出了两大核心创新。第一是“矩-阵值状态”。它将 RWKV 的递归状态从一个向量升级为一个矩-阵，可以被理解为一个被压缩了的、低秩的 KV 缓存。相比于只能存储混合信息的向量状态，矩-阵状态能够-更结构化地存储和区-分来自过去不同 token 的信息。第二是“动态递归”。作者引入了一个数据依赖的门控机制，让模型能够-根据当-前的输入 token 来动态地调整其状态矩-阵的衰减率和更新强度。这使得模型的记忆管理变得更加灵活和智慧。

主要贡献与发现：
1.  **增强的隐藏状态**：通过将隐藏状态从向量提升至矩-阵，极大地增强了模型在固定大小内存中存储历史信息的能力，是在 RNN 框架内对 Transformer 的 KV 缓存的一-种高效近似。
2.  **智慧的记忆管理**：动态递归机制赋予了模型内容感知的记忆更新能力，使其在处理长篇文档或对话时，能够-更好地捕捉长距离依赖关系。
3.  **水平循环的 SOTA**：这篇论文代表了“水平循环”潜-在推理方向的最新技术水平，为构-建能够-处理无限长序列的语言模型提供了一-条极具前景的技术路径。`
  },
  {
    id: "dao2024mamba",
    title: "Transformers are ssms: Generalized models and efficient algorithms through structured state space duality",
    authors: "Dao & Gu",
    year: 2024,
    method: "Mamba / Selective SSM",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "揭示并利用了 Transformer 注意力机制与结构化状态空间模型（SSM）的数学对偶性，提出“选择性 SSM”：其核心矩阵由当前输入动态生成，从而模仿注意力的内容感知能力。同时设计了高效的并行扫描算法，使其训练时可并行，推理时可切换为 O(1) 循环模式。",
    formula: "h_t = A_t h_{t-1} + B_t x_t; y_t = C_t h_t",
    contribution: `这篇论文是长序列建模领域的里程碑，深刻地揭示并利用了 Transformer 中的自注意力机制与结构化状态空间模型（S4）之间的数学对偶性，催生了 Mamba 架构。

核心思想与方法：论文的核心洞见在于证明了某些特定形式的注意力机制在数学上等价于一个 SSM 的输出。其关键突破在于提出了一-种“选择性 SSM”，它的核心矩-阵不再是固定的，而是由当-前的输入数据动态生成的。这个“选择性”机制模仿了注意力机制根据 query 和 key 计算权重，动态地决定从 value 中读取哪些信息。更重要的是，作者设计了一-种硬件感知的并行扫描算法，使得这种看似递归的选择性 SSM 可以在训练时像 Transformer 一样高效地并行计算，而在推理时又能切换回 RNN 模式，实现极速的、O(1) 内存占用的自回归生成。

主要贡献与发现：
1.  **统一 Transformer 与 SSM**：这篇论文在理论层面建立了一座桥梁，深刻地统一了注意力机制和状态空间模型。
2.  **Mamba 架构的诞生**：基于选择性 SSM 的思想，作者提出了 Mamba 架构。Mamba 在一系列语言建模和长序列理解任务上，达到了与 Transformer 相当甚至更好的性能，但其训练和推理速度都快得多。
3.  **水平循环的新范式**：Mamba 为“水平循环”潜-在推理提供了一-种全新的、极其强大的范式。它既有 RNN 的高效递归形式，又有类似注意力的动态选择能力，成为当-前处理超长序列任务的最主流和最有效的技术之一。`
  },
  {
    id: "sun2021retnet",
    title: "Retentive network: A successor to transformer for large language models",
    authors: "Sun et al.",
    year: 2021,
    method: "Retentive Network (RetNet)",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "提出创新的“回溯机制”，它有两种等价的计算形式：训练时可像注意力一样并行计算，推理时可转换为 RNN 的 O(1) 循环形式，从而在一个统一框架下同时实现“并行训练、高效推理、强大性能”的不可能三角。",
    formula: "\\text{Retention}(X) = (QK^T \\odot D)V",
    contribution: `这篇论文提出了“回溯网络”（RetNet），旨在解决 LLM 领域的一个核心“不可能三角”问题：一个理想的架构很难同时具备高效的并行训练、低成本的推理（O(1) 复杂-度）以及强大的性能。

核心思想与方法：RetNet 的核心在于其创新的“回溯机制”，它旨在替代标准的自注意力机制。回溯机制有两种等价的计算形式：
1.  **并行表示**：在训练时，回溯机制可以像自注意力一样，一次性地、并行地计算出序列中所有 token 之间的相互作用。它通过一个带有指数衰减的权重矩-阵来实现。
2.  **递归表示**：在推理时，同样的回溯机制可以被巧妙地转-换为一个 RNN 的形式。模型的状态可以被总结为一个固定大小的隐藏状态，在生成每一个新 token 时，只需用当-前的输入对这个状态进行一次简单-的更新即可。

这两种表示之间的数学等价性是 RetNet 的精髓所在。它使得模型能够在训练和推理时“无缝切换”，在不同的场景下选择最优-的计算模式。

主要贡献与发现：
1.  **打破不可能三角**：RetNet 是首批成功地在一个统一框架下，同时实现训练并行化、推理 O(1) 复杂-度和强大性能的架构之一。
2.  **回溯机制**：提出了一-种全新的、基于位置衰减的序列交互建模机制，可以被视-为自注意力的一-种高效替代品。
3.  **水平循环的重要范式**：RetNet 为“水平循环”潜-在推理提供了一-种独特且优-雅的实现。它的递归形式使得模型能够-理论上处理无限长的序列，而其背后的并行形式又保证了模型的可训练性。`
  },
  {
    id: "lan2025liger",
    title: "Linearizing large language models to gated recurrent structures (Liger)",
    authors: "Lan et al.",
    year: 2025,
    method: "Liger",
    category: "Latent Reasoning Paradigms",
    influences: ["sun2021retnet"],
    core_idea: "提出一种低成本将预训练 Transformer 模型“线性化”为门控循环结构的方法。它巧妙地重新利用原注意力机制中的 Key 矩阵，将其功能转变为循环单元中的“遗忘门”参数，从而在几乎不增加训练成本的情况下，将模型的推理模式变为高效的 O(1) 循环模式。",
    formula: "f_t = \\sigma(W_f h_{t-1} + U_f x_t); h_t = (1-f_t) \\odot h_{t-1} + \\dots",
    contribution: `这篇论文探索了一条与从头训练高效长序列模型截然不同的路径：将一个已经预训练好的、强大的 Transformer 模型，“转-化”或“线性化”为一个高效的门控循环结构。

核心思想与方法：论文提出的方法名-为 Liger。它的目标是将 Transformer 中的自注意力块，替-换为一个参数高效的、功能近似的门控循环单元。其巧妙之处在于，它并非引入全新的参数，而是“重新利用”了原 Transformer 模型中预训练好的权重。具体来说，Liger 将原注意力机制中的 Key (K) 矩-阵，重新诠释为一个用于计算循环单元中“遗忘门”的参数。这个遗忘门会根据当-前的输入，动态地决定应该在多大程度上保留过去的隐藏状态。这个设计基于一个直观的假设：在自注意力中，一个 query 和一个 key 的点积较小，意味-着这个 key 对应-的过去信息不那么重要，这在循环机制中就对应-于一个较高的“遗忘”概率。

主要贡献与发现：
1.  **低成本的模型转-换**：Liger 提出了一-种极其高效的方法，将昂贵的 Transformer 推理转-换为廉价的 RNN 推理，而训练成本极低（仅需原预训练成本的 0.02%）。
2.  **权重再利用**：该工作展示了预训练模型权重的高度可塑性，证明了注意力机制中学习到的模式可以被成功地迁移和再利用于构-建循环动态。
3.  **实用化的水平循环**：这项工作代表了“模型转-换”这一实用化技术路线的最新进展，使得社-区能够-快速地将现有的、强大的开源 Transformer 模型，改造成能够-处理长序列、部署成本更低的循环模型。`
  },

  // =================================================================================
  // 第五部分：Interpretability (Mechanistic Interpretability)
  // =================================================================================
  {
    id: "wang2022circuit",
    title: "Interpretability in the wild: a circuit for indirect object identification in gpt-2 small",
    authors: "Wang et al.",
    year: 2022,
    method: "Circuit",
    category: "Interpretability",
    influences: ["geva2020ffn"],
    core_idea: "通过“逆向工程”的方法，在 GPT-2 small 模型中精确定位并验证了一个负责执行“间接宾语识别”这一复杂语法功能的、由多个注意力头协同工作的完整“计算回路”，揭示了模型内部组件如何自发学习到高度专门化的算法功能。",
    formula: "\\text{Output} = \\text{NameMoverHeads}(\\text{S-InhibitionHeads}(\\text{Input}))",
    contribution: `这篇论文是Interpretability领域的典范之作，它展示了如何通过“逆向工程”的方法，在一个预训练好的模型中，精-确地识别出一个负责执行特定语法功能的、可解释的“计算回路”。

核心思想与方法：研究者选择的任-务是“间接宾语识别”。团队采用了一-种迭代式的“发现-验证”方法：
1.  **假设与追踪**：他们首先假设模型内-部存在特定的注意力头来执行特定功能，例如“名-称移动头”，负责将主语或宾语的名-字从早期位置复制到当-前位置。
2.  **回路发现**：通过不断地追踪信息流，他们发现了一个由多个不同功能的注意力头组成的复杂回路，包括负责排除干扰项的“S-抑制头”和负责复制信息的“名-称移动头”。
3.  **因果验证**：为了证明他们发现的回路确实是模型执行此任-务的核心机制，他们进行了精-确的“回路修补”实验，通过“嫁接”回路的输出来验证其功能。

主要贡献与发现：
1.  **找到了具体的算法**：这是首次在大型语言模型中如此精-确地识别出-一个完整的、可解释的、执行自然语言处理任-务的算法回路。
2.  **证明了功能专业化**：研究表明，模型中的不同组件会自发地学习到高度专业化的功能，并像计算机程序中的模组一样协同工作。
3.  **对潜-在推理的启示**：这项工作为理解“潜-在 CoT”提供了微观层-面的证据。它揭示了多步推理中的每一步都可能对应-到模型中一个具体的子回路。模型的层级堆叠，正是为这些回路的顺序执行和信息整合提供了计算基础。`
  },
  {
    id: "skean2025layer",
    title: "Layer by layer: Uncovering hidden representations in language models",
    authors: "Skean et al.",
    year: 2025,
    method: "Linear Probing Analysis",
    category: "Interpretability",
    influences: ["wang2024matching"],
    core_idea: "系统性地分析了 LLM 所有中间层的表征能力，发现对于大多数下游任务，模型的最优表征并非在最后-一层，而是在中间层（约 2/3 深度处）。这证明了明显的层级功能分工：浅层处理句法，中层形成丰富语义，深层负责决策，为“层级是潜-在 CoT 基本单位”的观点提供了宏观证据。",
    formula: "\\text{Performance}_l = \\text{TrainLinearProbe}(\\text{Embeddings}_l)",
    contribution: `这篇论文深入探讨了 LLM 内-部表征的层级化结构，旨在回答一个核心问题：信息是如何在 Transformer 的各个层之间进行处理、转-换和提炼的？

核心思想与方法：研究者提出了一-种全面的评-估框架，用于探-测和比较 LLM 不同层级隐藏状态的质量。他们从模型的每一层提取出 token 的隐藏表征，然后在这些表征之上训练简单-的线性探-测器，去完成一系列下游的 NLP 任-务。线性探-测器的性能被用来衡量该层表征所包含的、与任-务相关的信息的“可读取性”。

主要贡献与发现：
1.  **中间层的表征优-势**：一个惊人的核心发现是，对于大多数任-务而言，模型的最优-表征并不存在于最后-一层，而是出现在其中间层。这表明，模型在深层的处理中，为了专注-于“预测下-一个词”，可能会丢弃掉一些对下游任-务有用的、更泛化的语义信息。
2.  **层级功能分化**：研究证实了明显的层级功能分工。浅层侧重于捕捉句法信息；中间层逐渐形成丰富的、高度抽象的语义表征；而深层则更像是“决策层”，负责整合和转-换信息。
3.  **潜-在 CoT 的宏观证据**：这项工作为《A Survey on Latent Reasoning》中的“分层专业化理论”提供了强有力的宏观证据。它清晰地展示了 Transformer 的层级堆叠结构本身就是一个信息处理的流水线，一个隐式的计算链。`
  },
  {
    id: "geva2020ffn",
    title: "Transformer feed-forward layers are key-value memories",
    authors: "Geva et al.",
    year: 2020,
    method: "FFN as Key-Value Memory",
    category: "Interpretability",
    influences: [],
    core_idea: "首次对 Transformer 中神秘的前馈网络（FFN）层的功能提出一个具体的、可解释的模型：FFN 在功能上等价于一个“键值内存”。其第-一层充当“键”，用于匹配输入模式；第二层充当“值”，用于检索并输出与匹配模式相关联的知识。",
    formula: "\\text{FFN}(x) = \\text{ReLU}(xW_1)W_2",
    contribution: `这篇论文是Interpretability领域的奠基之作，它首次对 Transformer 模型中看似“黑箱”的前馈神-经网络（FFN）层的功能，提出了一个具体的、可解释的计算模型。

核心思想与方法：论文提出了一个惊人但直观的假设：FFN 层在功能上等价于一个“键值内存”。
*   **第-一个线性层 (W1)**：充当“键”矩-阵。当-一个输入 token 的隐藏表示 x 输入时，它会与 W1 的每一行（代表一个模式或概念）进行点积运-算，以“匹配”模式。
*   **非线性激活 (如 ReLU)**：充当一个过滤器，只允许被成功匹配的模式的信号通过。
*   **第二-个线性层 (W2)**：充当“值”矩-阵。W2 的每一列对应-于 W1 中一个模式的“输出”。激活函数的输出会加权地组合 W2 的各列，从而将匹配到的模式所对应-的“值”写入到最终的输出中。

简而言之，FFN 层的工作流程是：用输入去匹配一系列预存的模式（键），然后检索并输出与最匹配模式相关联的信息（值）。

主要贡献与发现：
1.  **FFN 的可解释模型**：该工作为理解 FFN 层的功能提供了一个强有力的、简洁的隐喻和计算模型，将其从一个纯粹的非线性变换器，变为一个可查询的、结构化的记忆系统。
2.  **知识定位**：这个模型意味-着，语言模型中相当一部分的参数化知识可能就存储在 FFN 层的权重中，后续的知识编辑研究都建立在这一发现之上。
3.  **潜-在推理的基-础**：在一个多步推理过程中，自注意力层负责路由信息，而 FFN 层则负责根据传递来的信息，触发相关的知识和规则，生成新的中间结果。`
  },
  {
    id: "wang2024matching",
    title: "Towards understanding how transformer perform multi-step reasoning with matching operation",
    authors: "Wang et al.",
    year: 2024,
    method: "Matching Operation Analysis",
    category: "Interpretability",
    influences: [],
    core_idea: "提出 Transformer 执行多跳推理的核心计算单元是“匹配操作”：在推理链的每一步，模型都通过注意力头和 FFN 层的协同，执行一次“查询-匹配-传播”的过程。层数的增加使得模型能级联执行多次匹配，并并行探索呈指数级增长的推理路径。",
    formula: "h_{l+1} = \\text{Match}_{l+1}(\\text{Query}(h_l))",
    contribution: `这篇论文深入探究了 Transformer 模型执行多跳（multi-hop）推理的内-部机制，这-种推理要求模型能够-将多个分散在上下文中的独立事实联系起来，形成一条逻辑链。

核心思想与方法：论文提出，Transformer 执行多跳推理的核心计算单元是一-种“匹配操作”。作者假设，在推理链的每一步，模型都需要执行一次“查询-匹配-传播”的过程。例如，回答“提出相对论的人在哪-个国家出生？”需要模型首先将“相对论”作为查询，匹配到“爱因斯坦”，然后将“爱因斯坦”作为新查询，匹配到“德国”。研究者证明，这种级联的匹配操作可以由 Transformer 的层级结构自然地实现。特定的注意力头和 FFN 层会协同工作，形成专门的“匹配回路”。

主要贡献与发现：
1.  **多跳推理的机制模型**：论文为 Transformer 执行多跳推理的过程提供了一个具体的、基于“匹配操作”的机制模型，将宏观的推理行为与微观的网络计算联系起来。
2.  **揭示深度的作用**：研究清晰地阐释了为什么增加模型深度（层数）能够-极大地增强其多跳推理能力。每一层都为推理链增加了一个潜-在的步骤，而层的堆叠使得并行探索复杂推理图成为可能。
3.  **对潜-在 CoT 的精细化理解**：这篇论文揭示了潜-在推理回路在执行一个具体的、复杂的推理任-务时的具体算法结构，为我们理解潜-在 CoT 中每一步的计算内容提供了更精细的视角。`
  },
  {
    id: "sun2025cursedepth",
    title: "The curse of depth in large language models",
    authors: "Sun et al.",
    year: 2025,
    method: "Theoretical & Empirical Analysis",
    category: "Interpretability",
    influences: ["dehghani2018ut"],
    core_idea: "系统性地揭示了“深度的诅咒”：在采用 Pre-LayerNorm 的 Transformer 中，当深度超过一定限度，模型性能会因“输出方差指数爆炸”和“注意力矩-阵退化”两大问题而急剧下降，这对依赖增加计算深度来实现潜-在推理的方向提出了严峻挑战。",
    formula: "\\text{Var}(x^L) \\propto \\exp(\\alpha L)",
    contribution: `这篇论文对“深度是 LLM 推理能力的关键”这一普遍观点，提出了一个重要的警告和深刻的理论分析，揭示了“深度的诅咒”。

核心思想与方法：论文从理论和实验两个层-面，剖析了导致深度诅咒的 两大核心机制，特别是在采用了 Pre-LayerNorm 结构的现代 Transformer 中：
1.  **输出方差的指数爆炸**：研究者从理论上证明，在 Pre-LN Transformer 中，随着层数的增加，模型输出的方差会呈指数级增长，导致模型的输出变得极-其不稳定，使得深层模型的训练异常困难。
2.  **注意力矩-阵的退化**：论文通过实验观察发现，在非常深的模型中，其深层的注意力头的注意力模式会出现“退化”现象，坍缩成一个近乎“秩一”的矩-阵，即所有的查询都只关注极少数几个键。这意味-着深层的注意力机制几乎完全失效。

主要贡献与发现：
1.  **揭示深度的局限性**：该工作首次系统性地揭示并解释了深度在 LLM 中的负面效应，为模型设计提供了重要的理论指导。
2.  **对垂直循环的警示**：这篇论文直接对“垂直循环”潜-在推理提出了挑战。如果简单-地让模型在同一组层上反覆迭代，很可能会迅速遭遇“深度的诅咒”。这意味-着成功的循环架构必须包含一些特殊设计来保持计算的稳定性。
3.  **推动架构创新**：这-项研究激-励了后续一系列旨在解决深度模型训练稳定性问题的工作。`
  },
  
  // =================================================================================
  // 第六部分：Latent Reasoning Paradigms (Spatial Infinite Reasoning)
  // =================================================================================
  {
    id: "ye2024dot",
    title: "Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models",
    authors: "Ye et al.",
    year: 2024,
    method: "Diffusion of Thoughts (DoT)",
    category: "Latent Reasoning Paradigms",
    influences: ["wei2022cot"],
    core_idea: "将 CoT 推理与扩散模型相结合，将完整的推理链（包括所有中间步骤）视为一个整体。模型从一个充满掩码的模板开始，在多个去噪步骤中，并行地、迭代式地填充和修正整个推理过程，从而实现全局规划和逻辑自洽。",
    formula: "p_{\\theta}(x_{0:T}) = p(x_T) \\prod_{t=1}^{T} p_{\\theta}(x_{t-1} | x_t)",
    contribution: `这篇论文巧妙地将扩散模型的迭代精炼特性与 CoT 推理相结合，提出了“思维扩散”（Diffusion of Thoughts, DoT）。传统自回归模型逐字生成，无法全局规划。

核心思想与方法：在 DoT 框架中，一个完整的推理链被视-为一个整体。模型从一个充满“未知”掩码（mask）或噪-声的模板开始，然后在多个去噪步骤中，同时地、并行地填充和修正整个推理过程。例如，模型可能先生成一个粗略的推理框架，然后在后续步骤中填充细节，并修正早期步骤中与后期结论不符的逻辑错误。在每个去噪步骤中，模型都能看到整个序列的（带噪-）表示，这使得它能够-基于全局上下文进行决策。

主要贡献与发现：
1.  **并行的 CoT 生成**：DoT 将 CoT 从一个线性的、不可逆的过程，转-变为一个并行的、可修正的全局优化过程。
2.  **实现无限深度推理**：这种迭代式的全局精炼过程，正是“无限深度推理”的体现。模型的“思考时-间”或“推理深度”直接对应-于去噪的步数。在推理时，可以通过增加去噪步数来投入更多的计算，让模型对解决方案进行更深入的打磨和修正。
3.  **提升复杂推理能力**：该框架使得模型能够-处理那些需要反覆权衡和自我校正的复杂推理任-务，例如规划、数学证明和程序代码生成，为Latent Reasoning Paradigms提供了坚实的技术基-础。`
  },
  {
    id: "ma2025dkvcache",
    title: "dkv-cache: The cache for diffusion language models",
    authors: "Ma et al.",
    year: 2025,
    method: "dKV-Cache",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "为解决扩散模型推理成本高昂的问题，提出一种“延迟和条件化”的 KV 缓存策略。在每一步去噪中，只为那些预测置信度超过阈值的“已确定” token 计算并缓存其 Key/Value，从而避免大量冗余计算，实现 2-10 倍的推理加速。",
    formula: "\\text{Cache}(K_t, V_t) \\text{ if } \\text{conf}(x_t) > \\tau",
    contribution: `这篇论文解决了基于扩散的语言模型在实际应用中一个非常棘手的工程和效率问题：如何在迭代去噪的过程中高效地利用 Transformer 的 KV 缓存机制。

核心思想与方法：论文提出了一-种名-为“dKV-Cache”的巧妙缓存策略。其核心思想是，在扩散模型的迭代过程中，序列中的 token 并非以相同的速-度变得“确定”。因此，没有必要在每一步都为所有 token 重新计算它们的 Key 和 Value。dKV-Cache 引入了一-种“延迟和条件化”的缓存更新机制：
1.  **置信度评-估**：在每一步去噪之后，模型会评-估序列中每个 token 的预测置信度。
2.  **选择性缓存**：只有那些置信度超过某个阈值的“已确定” token，它们的 Key 和 Value 才会被计算并存入 dKV-Cache 中。
3.  **条件化计算**：在下-一步去噪时，对于那些置信度较低的“未确定” token，它们的注意力计算会同时利用来-自 dKV-Cache 中已确定 token 的 Key/Value。

主要贡献与发现：
1.  **加速扩散模型推理**：dKV-Cache 为扩散语言模型的推理提供了显著的加速，使其在实际应用中变得更加可行。
2.  **弥合扩散与自回归的鸿沟**：这项工作巧妙地将自回归模型中成熟的 KV 缓存技术，适配到了非自回归的扩散模型框架中。
3.  **对无限深度推理的实用化**：虽然扩散模型理论上可以进行无限步的精炼，但高昂的计算成本是其主要障碍。dKV-Cache 这样的效率优-化技术，是将“无限深度推理”从一个理论概念推向实用化工具的关键一步。`
  },
  {
    id: "sahoo2024simplemdm",
    title: "Simple and effective masked diffusion language models",
    authors: "Sahoo et al.",
    year: 2024,
    method: "Simple-MDM",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "证明了训练强大的扩散语言模型无需复杂的噪-声方案和损失函数，一个类似 BERT 预训练的、极简的“随机掩码-预测”方案就已足-够。该方法可直接利用现有的预训练编码器，极大地降低了技术门槛。",
    formula: "\\mathcal{L} = \\text{CrossEntropy}(p_{\\theta}(x_{\\text{masked}}), x_{\\text{original}})",
    contribution: `这篇论文致力于简化和改进基于掩码的扩散语言模型（MDMs），使其训练更稳定、更高效。早期的离散扩散模型通常需要复杂的噪-声方案和损失函数，实现和调优相对困难。

核心思想与方法：论文提出的“简单- MDM”的核心在于其极简的设计。其训练过程如下：
1.  **随机掩码**：取一个干净的文本序列，随机选择一定比例的 token，用一个特殊的 \`[MASK]\` 标-记替-换它们。
2.  **预测被掩码的 token**：将这个带有掩码的序列输入到一个 Transformer 模型中，任-务就是预测出所有被 \`[MASK]\` 的位置上原始的 token 是什么。
3.  **简单-的交叉熵损失**：训练的损失函数就是标准的交叉熵损失。

在推理时，模型从一个完全由 \`[MASK]\` 组成的序列开始，然后进行多步迭代。在每一步中，模型都会对所有 \`[MASK]\` 位置给出预测，然后根据置信度，选择一部分 \`[MASK]\` 标-记用预测结果来替-换，直到生成一个完整的句子。

主要贡献与发现：
1.  **简化 MDM 训练**：论文证明了训练强大的扩散语言模型并不需要复杂的理论框架，极大地降低了该技术路线的门槛。
2.  **高效利用编码器**：该方法可以直接利用现有的、强大的预训练编码器模型（如 RoBERTa），只需对其进行生成式微调即可。
3.  **空间推理的基-础**：Simple-MDM 是“Latent Reasoning Paradigms”的一个基-础且核心的范例。它的迭代式 unmasking 过程，就是一个在全局上下文下，对整个序列进行逐步求精和完善的过程。`
  },
  {
    id: "ye2025dream",
    title: "Dream 7b: Unifying diffusion and autoregressive generation",
    authors: "Ye et al.",
    year: 2025,
    method: "Dream (Hybrid AR-Diffusion)",
    category: "Latent Reasoning Paradigms",
    influences: ["ye2024dot"],
    core_idea: "提出一种“AR-初始化，Diffusion-精炼”的混合生成框架。首先，一个高效的自回归模型快速生成一份草稿；然后，一个扩散模型从这份“带少量噪-声的草稿”开始进行多轮迭代精炼，修复其中的全局逻辑错误。该方法兼顾了 AR 的局部流畅性与 Diffusion 的全局规划能力。",
    formula: "x_0 \\sim p_{\\text{AR}}(x); \\quad x_{t-1} \\sim p_{\\text{Diffusion}}(x_t, x_0)",
    contribution: `这篇论文旨在融合自回归（AR）模型和扩散模型的优-点，提出了一-种名-为 Dream 的混合式生成框架。AR 模型的优-点在于局部流畅性好，而扩散模型的优-点在于强大的全局规划和修改能力。

核心思想与方法：Dream 框架的核心是一-种“AR-初始化，Diffusion-精炼”的混合策略。
1.  **自回归初始化**：首先，一个高效的 AR 模型会快速地生成一个草稿。这个草稿为整个生成提供了一个良好且流畅的起点。
2.  **扩散式精炼**：接着，这个草稿会被输入到一个扩散模型中。扩散模型并非从纯噪-声开始，而是从这个“带有少量噪-声的草稿”开始进行去噪。在迭代的去噪过程中，扩散模型会利用其全局视野，对草稿进行多轮的修改和润色。

此外，Dream 还引入了“上下文自适应噪-声调度”，即根据 AR 模型生成草稿时的置信度，来决定在精炼阶段对不同部分加入多少噪-声。

主要贡献与发现：
1.  **混合 AR 与扩散**：Dream 成功地将两种主流生成范式有机地结合在一起，取长补短，创造出-一个在性能、速-度和可控性上都表现出色的新框架。
2.  **提升生成质量和效率**：相比于纯扩散模型，从一个高质量的草稿开始可以显著减少所需的去噪步数。相比于纯 AR 模型，后续的扩散精炼步骤极大地提升了生成文本的全局质量和逻辑一致性。
3.  **空间推理的实用化演进**：混合式模型为解决复杂的、需要兼顾全局规划和局部细节的生成任-务提供了更强大、更灵活的工具。`
  },
  {
    id: "zhu2025llada",
    title: "Llada 1.5: Variance-reduced preference optimization for large language diffusion models",
    authors: "Zhu et al.",
    year: 2025,
    method: "VRPO (Variance-Reduced Preference Optimization)",
    category: "Latent Reasoning Paradigms",
    influences: [],
    core_idea: "首次将人类偏好对齐技术成功应用于大型扩散语言模型。提出 VRPO 算法，通过“无偏蒙特卡洛预算分配”和“对偶采样”等方差缩减技术，解决了在扩散模型的随机生成轨迹上进行偏好学习时梯度估-计方差过大的难题，使其能有效学习生成更符合人类逻辑的推理过程。",
    formula: "\\mathcal{L}_{\\text{DPO-like}} = -\\mathbb{E} [\\log \\sigma(\\beta \\log \\frac{\\pi_{\\theta}(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_{\\theta}(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)})]",
    contribution: `这篇论文是将人类偏好对齐技术成功应用于大型扩散语言模型的关键工作。如何使其生成结果更符合人类的逻辑、价-值观和偏好，成为一个核心问题。

核心思想与方法：论文提出了 LLaDA 1.5 框架，其核心是一-种名-为 VRPO 的新算法。VRPO 旨在解决在扩散模型的随机生成轨迹上进行偏好学习时，梯度估-计方差过大的核心难-题。其流程大致如下：
1.  **偏好数据收集**：首先需要一个偏好数据-集，其中每个样本包含一个问题、一个人类偏好的“胜出”推理链和一个不太好的“落败”推理链。
2.  **方差缩减的目标函数**：VRPO 的关键创新在于其目标函数的设计。它引入了两种强大的方差缩减技术：“无偏蒙特卡洛预算分配”和“对偶采样”，从而能够-在复杂-的生成轨迹空间中，得到一个更稳定、更可靠的梯度信号，有效地引导扩散模型学习生成更符合人类偏好的推理过程。

主要贡献与发现：
1.  **扩散模型的偏好对齐**：这项工作首次为大型扩散语言模型提供了一个高效、稳定的偏好对齐框架，成功地将 DPO 的思想扩展到了非自回归的生成范式中。
2.  **提升推理质量**：实验表明，经过 VRPO 对齐的模型，其生成结果的准确性和逻辑性显著优-于仅经过监督微调的基-础模型。
3.  **空间推理的精细控制**：这项工作将“Latent Reasoning Paradigms”从“能生成”推向了“生成得好、生成得对”，为精细地控制扩散模型的推理行为、使其符合复杂-的人类规范提供了强有力的工具。`
  },

  // =================================================================================
  // 第七部分：Training & Optimization (The Optimization-Based Perspective)
  // =================================================================================
  {
    id: "sun2024ttt",
    title: "Learning to (learn at test time): Rnns with expressive hidden states (TTT)",
    authors: "Sun et al.",
    year: 2024,
    method: "Test-Time Training (TTT)",
    category: "Training & Optimization",
    influences: [],
    core_idea: "提出一种全新视角，将 RNN 的隐藏状态更新过程重新诠释为“测试时的线上优-化”。隐藏状态被视-为一组需要即时学习的“快速权重”，每处理一个新 token，模型就在执行一步梯度下降，以优-化一个隐式的目标函数，从而让状态能最好地“记住”新信息。这深刻地将“时-间深度”等价于“优-化迭代次-数”。",
    formula: "S_t = S_{t-1} - \\eta \\cdot \\nabla_S \\mathcal{L}(S_{t-1}; x_t)",
    contribution: `这篇论文为“水平循环”潜-在推理和“基于优-化的视角”提供了关键的理论和实践基-础。它旨在解决标准 Transformer 在处理超长序列时遇到的记忆和计算瓶颈。

核心思想与方法：论文提出了一-种全新的视角，即将 RNN 的隐藏状态更新过程重新诠释为一个“测试时的线上优-化”（Test-Time Training, TTT）过程。在这个框架下，模型的隐藏状态不再仅仅是一个被动的信息载体，而被视-为一组“快速权重”。当模型处理序列中的每一个新的 token 时，它并不仅仅是更新隐藏状态，而是在执行一步梯度下降，以优-化一个局部的、隐式的目标函数。这个目标函数旨在让当-前的隐藏状态能够-最好地“记住”或“拟合”刚观测到的新信息。整个处理长序列的过程，就等同于在测试时对这个隐藏状态“层”进行持续的、线上式的训练。

主要贡献与发现：
1.  **优-化视角下的长序列处理**：TTT 框架将 RNN 的序列处理与神-经网络的优-化过程深刻地联系在一起，揭示了“时-间上的深度”可以等价于“参数上的优-化次-数”。
2.  **高效且具表达力的 RNN**：基于 TTT 思想设计出的模型在保持 O(1) 内存占用的同时，展现了远超传统 RNN 的长距离依赖建模能力。
3.  **潜-在推理的理论统一**：这个“用时-间换深度”的观点将“水平循环”和“垂直循环”统一在“迭代优-化”这一共同的框架下，为实现“无限深度推理”提供了理论基-础。`
  },
  {
    id: "behrouz2025atlas",
    title: "Atlas: Learning to optimally memorize the context at test time",
    authors: "Behrouz et al.",
    year: 2025,
    method: "Atlas with Muon Optimizer",
    category: "Training & Optimization",
    influences: ["sun2024ttt"],
    core_idea: "在 TTT 框架基-础上，引入更先进的二阶优-化器（近似牛顿法），以更高效地更新模型的循环隐藏状态。通过一种名为 Muon 的高效二阶优-化算法，Atlas 在处理每个 token 时能比一阶方法“学得更快、记得更牢”，在百万级 token 的“大海捞针”任-务上取得了 SOTA 性能。",
    formula: "S_t = S_{t-1} - \\eta \\cdot H_{t-1}^{-1} \\nabla_S \\mathcal{L}(S_{t-1}; x_t)",
    contribution: `这篇论文是基于优-化视角的长序列处理（即 TTT 框架）的又一重要进展，它引入了更先进、更强大的二阶优-化器，以进一步提升模型在测试时记忆和利用长上下文的能力。

核心思想与方法：论文的核心贡献是设计并实现了一-种高效的、近似的二阶优-化算法，用于在测试时更新模型的递归隐藏状态。这个算法被名-为 Muon。作者们采用了一-种名-为 Newton-Schulz 迭代的方法来近似 Hessian 的逆，这种方法只需要一系列的矩-阵乘法，计算效率远高于直接求逆。搭载了 Muon 优-化器的模型被名-为 Atlas。在处理长序列时，Atlas 的隐藏状态更新不再是简单-的一阶梯度步进，而是执行一步近似的牛顿法更新。

主要贡献与发现：
1.  **将二阶优-化引入 TTT**：Atlas 首次成功地将高效的二阶优-化器应用于测试时的隐藏状态学习，显著提升了模型在处理每个 token 时的记忆效率和准确性。
2.  **SOTA 长序列性能**：实验结果显示，Atlas 在多项极具挑战性的长上下文“大海捞针”任-务上取得了当-时最先进的性能。
3.  **深化“时-间换深度”的理解**：如果说 TTT 建立了“处理一个 token = 执行一步优-化”的对应-关系，那么 Atlas 则进一步证明了“优-化步骤的质量”至关重要。使用更强的优-化器，相当于在相同的“时-间”内，获得了“更深”的有效计算深度。`
  },
  {
    id: "schone2025implicit",
    title: "Implicit language models are RNNs: Balancing parallelization and expressivity",
    authors: "Schöne et al.",
    year: 2025,
    method: "Implicit Fixed-Point RNNs",
    category: "Training & Optimization",
    influences: ["dao2024mamba", "sun2024ttt"],
    core_idea: "从“隐式层”视角，将 RNN 的状态更新视为求解一个不动点方程。证明了通过对一个简单的线性状态空间块（如 Mamba）进行迭代直至收敛，其最终的转-移函数等价于一个功能强大的非线性 RNN。该方法在训练时可并行，在推理时只需少量迭代即可收敛，兼顾了并行性与表达力。",
    formula: "h_t = \\text{FixedPoint}(\\text{SSMBlock}, h_{t-1}, x_t)",
    contribution: `这篇论文从一个独特的“隐式层”视角，重新审视和连接了现代的 SSM 与经典的 RNN。它旨在解决一个长期存在的矛盾：传统 RNN 表达能力强但无法并行训练；现代 SSM 可并行训练但核心是线性的。

核心思想与方法：论文提出，可以将一个 RNN 的单步状态更新看作是一个“隐式方程”的求解过程，即寻找一个不动点。作者证明了，通过对一个简单-的线性状态空间块进行迭代，直到其隐藏状态收敛，所得到的最终状态转-移函数在数学上等价于一个功能强大的、非线性的 RNN。这个过程被名-为“隐式不动点 RNN”。在训练时，可以利用隐式微分等技术高效地并行计算梯度；在推理时，模型只需对每个 token 执行少量内-部迭代即可达到收敛。

主要贡献与发现：
1.  **统一 SSM 与非线性 RNN**：这篇论文在理论上深刻地统一了可并行训练的线性 SSM 与表达能力更强的非线性 RNN。
2.  **兼顾并行性与表达力**：Implicit Fixed-Point RNNs 提供了一-种新的模型设计范式，它既像 SSM 一样可以高效并行训练，又具备传统 RNN 的非线性动态建模能力。
3.  **另一-种形式的无限深度**：这项工作为“无限深度推理”提供了一-种独特的视角。这里的“深度”是内-部不动点迭代的次-数，模型可以自适应地决定每个 token 需要多少次迭代才能达到稳定的内-部表示，是一-种发生在单个时-间步内-部的、深度的潜-在推理过程。`
  },
  {
    id: "munkhdalai2024infini",
    title: "Leave no context behind: Efficient infinite context transformers with infini-attention",
    authors: "Munkhdalai et al.",
    year: 2024,
    method: "Infini-attention",
    category: "Training & Optimization",
    influences: ["sun2024ttt"],
    core_idea: "提出一种在标准 Transformer 中直接集成长期记忆的机制。在每个注意力层，模型并行地对近期上下文执行标准注意力，同时用一个线性的、类似 RNN 的规则去更新一个代表了遥远历史的“压缩记忆体”。最终输出是两者的结合，实现了对 Transformer 的微创、高效长序列扩展。",
    formula: "O_t = \\text{Attn}(Q_t, K_{<t}, V_{<t}) + \\text{Retrieve}(Q_t, S_{t-1})",
    contribution: `这篇论文提出了一-种名-为“无限注意力”的新机制，旨在让标准的 Transformer 模型能够-在不牺牲其核心架构的前提下，处理无限长的上下文。

核心思想与方法：Infini-attention 的核心是在标准的自注意力模组旁边，并联了一个“压缩内存”。其工作流程如下：
1.  **标准注意力**：在每个 Transformer 层，模型首先对最近的一段上下文执行标准的自注意力计算。
2.  **内存更新**：与此同-时，对于更早的、超出局部窗口的历史上下文，它们的 Key, Value 和 Query 会被用来更新一个长期记忆状态 S。
3.  **内存检索与整合**：在计算最终的注意力输出时，模型不仅会利用来-自局部窗口的信息，还会从长期记忆 S 中检索相关的历史信息。

通过这种“局部全注意 + 全局压缩记忆”的混合模式，Infini-attention 使得 Transformer 能够-在处理新的 token 时，既能精-确地关注近期上下文，又能高效地从无限长的历史中提取相关信息。

主要贡献与发现：
1.  **对 Transformer 的微创扩展**：Infini-attention 提供了一-种非常有吸引力的长序列扩展方案，因为它可以作为一-个插件来扩展现有预训练 Transformer 的上下文处理能力。
2.  **混合记忆模型**：该工作是混合记忆系统的一个典型范例，它清晰地分离了短期工作记忆和长期压缩记忆。
3.  **另一-种优-化视角**：Infini-attention 的长期记忆更新过程也可以被看作是一-种线上优-化。每一次更新，都是用新的观测数据去“优-化”那个代表了全部历史的记忆状态 S。`
  },
  {
    id: "zhang2025lact",
    title: "Test-time training done right (LaCT)",
    authors: "Zhang et al.",
    year: 2025,
    method: "Large Chunk TTT (LaCT)",
    category: "Training & Optimization",
    influences: ["sun2024ttt"],
    core_idea: "对 TTT 框架的重大效率改进。针对原始 TTT 逐 token 更新导致的 GPU 利用率低下问题，LaCT 提出“大块测试时训练”：将输入序列划分为数千到百万 token 的大块，在块内-部并行执行多轮优-化迭代，充分利用 GPU 并行能力，从而在单卡上高效处理百万级 token 上下文。",
    formula: "S_{\\text{chunk}, t+1} = \\text{Optimizer}(S_{\\text{chunk}, t}, \\nabla \\mathcal{L}(X_{\\text{chunk}}))",
    contribution: `这篇论文是对 TTT 框架的一次深刻反思和重大改进。原始的 TTT 采用了一-种“小步快跑”的策略，在现代硬件上是极-其低效的。

核心思想与方法：为了克服这一瓶颈，论文提出了“大块测试时训练”（LaCT）。其核心思想与原始 TTT 截然相反，主张“积攒起来一起办”：
1.  **分块处理**：模型不再逐个 token 地更新其隐藏状态，而是将输入序列划分为非常大的块。
2.  **块内-并行处理**：在处理一个大块时，模型会在这-个大块上并行地执行多轮优-化迭代，以更新其隐藏状态，从而实现极高的硬件利用率。
3.  **块间递归**：一个块经过充分优-化后得到的最终隐藏状态，会作为处理下-一个块时的初始状态。

简而言之，LaCT 将 TTT 从一个“线上学习”的范式，转-变为一个“小批量学习”的范式。

主要贡献与发现：
1.  **解决 TTT 的效率瓶颈**：LaCT 通过分块处理，极大地提升了 TTT 框架的计算效率和硬件利用率。
2.  **扩展状态容量**：由于计算效率的提升，LaCT 使得使用更大规模的隐藏状态成为可能，这意味-着更强的记忆能力。
3.  **优-化视角的实用化**：这篇论文是将“基于优-化的视角”从一个理论框架推向大规模、实用化应用的关键一步，为未来设计更高效、更强大的长序列潜-在推理模型提供了重要的工程指导。`
  },

  // =================================================================================
  // 第八部分：Training & Optimization (Training Paradigm Innovation)
  // =================================================================================
  {
    id: "dong2025reinforcement",
    title: "Reinforcement pre-training",
    authors: "Dong et al.",
    year: 2025,
    method: "Reinforcement Pre-training",
    category: "Training & Optimization",
    influences: [],
    core_idea: "提出一种颠覆性的“强化预训练”范式，将传统的“预测下-一个词”任-务重构为一个强化学习问题。每预测一个词被视为一个“动作”，若预测正确则获得奖励。模型通过 actor-critic 算法学习最大化长期奖励，从而在预训练阶段就内-在地注入了目标导向的规划能力，而不只是被动模仿。",
    formula: "J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} [R(\\tau)]",
    contribution: `这篇论文提出了一-种颠覆性的训练范式——“强化预训练”，旨在从语言模型预训练的最底层-阶段就注入目标导向的推理能力。

核心思想与方法：该工作的核心是将“预测下-一个词”任-务从一个监督学习问题，重构成-为一个强化学习（RL）问题。在这个新框架下，每预测一个 token 不再仅仅是与标-签进行比较，而是被视-为一个“动作”，这个动作会立即得到一个“环境”给予的奖励（例如，预测正确奖-励为+1）。整个预训练过程变成了一个巨大的序列决策任-务。模型不仅要学习预测，更要学习一套能最大化长期累积奖-励的“策略”。作者引入了 actor-critic 算法，其中一个“评论家”网络被训练来评-估当-前上下文的价-值，这个价-值信号会指导模型（actor）的学习。

主要贡献与发现：
1.  **预训练范式的转-变**：这是首次将强化学习成功地、大规模地应用于语言模型的预训练阶段。它为模型注入了一-种内-在的“目标感”，从“看起来像”转-变为“做对事”。
2.  **提升推理与事实性**：实验表明，经过强化预训练的模型，在需要多步推理和高度事实准确性的任-务上，表现显著优-于使用传统方法训练的同等规模模型。
3.  **潜-在推理的早期培养**：这种训练方式可以被看作是在培养一-种“潜-在的规划能力”。为了最大化奖-励，模型必须在内-部隐式地评-估不同生成路径的潜-在价-值，这本身就是一-种复杂-的潜-在推理。`
  }
];

// Formulas have been converted to valid LaTeX format
const multimodal = [
  // =================================================================================
  // 第一部分：Core Models & Architectures与核心架构 (Foundational Works & Core Architectures)
  // =================================================================================
  {
    id: "vaswani2017attention",
    title: "Attention Is All You Need",
    authors: "Vaswani et al.",
    year: 2017,
    method: "Transformer",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "提出完全基于自注意力机制的 Transformer 架构，摒弃了循环和卷积，实现了高效的并行计算，并能捕捉长距离依赖关系。",
    formula: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
    contribution: "作为现代深度学习的基石，Transformer 架构不仅革新了自然语言处理，也为后续所有视觉-语言大模型（如ViT, CLIP, BERT）提供了核心的计算框架，是多模态领域发展的技术奇点。"
  },
  {
    id: "devlin2019bert",
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    authors: "Devlin et al.",
    year: 2019,
    method: "BERT / Masked Language Model",
    category: "Core Models & Architectures",
    influences: ["vaswani2017attention"],
    core_idea: "通过掩码语言模型（MLM）和下一句预测（NSP）两大任务，首次实现了基于 Transformer 的深度双向语言表示预训练，学习到了强大的上下文相关表示。",
    formula: "\\mathcal{L}_{\\text{MLM}} + \\mathcal{L}_{\\text{NSP}}",
    contribution: "开创了“预训练-微调”的范式，极大地推动了NLP的发展。其掩码建模思想被后续的视觉（BEiT）和多模态模型广泛借鉴，是构建强大编码器的核心技术之一。"
  },
  {
    id: "dosovitskiy2020vit",
    title: "An image is worth 16x16 words: Transformers for image recognition at scale",
    authors: "Dosovitskiy et al.",
    year: 2020,
    method: "Vision Transformer (ViT)",
    category: "Core Models & Architectures",
    influences: ["vaswani2017attention"],
    core_idea: "首次证明纯 Transformer 架构可以直接应用于图像识别，通过将图像分割成块（Patches）并将其视为词元序列，在超大规模数据集上预训练后性能超越了CNN。",
    formula: "z = [x_{\\text{class}}; E(p_1); E(p_2); \\dots; E(p_N)] + E_{\\text{pos}}",
    contribution: "打破了CNN在计算机视觉领域的统治地位，为后续的视觉-语言模型（VLMs）采用统一的 Transformer 架构作为视觉和文本主干网络铺平了道路，是多模态架构统一的里程碑。"
  },
  {
    id: "kim2021vilt",
    title: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision",
    authors: "Kim et al.",
    year: 2021,
    method: "ViLT (Vision-and-Language Transformer)",
    category: "Core Models & Architectures",
    influences: ["dosovitskiy2020vit"],
    core_idea: "提出一种极简的“单塔”视觉-语言模型，摒弃了复杂的物体检测器，直接将图像块嵌入和文本词元送入同一个 Transformer 进行端到端的深度交互。",
    formula: "\\text{Input} = [\\text{CLS}; W_e(T); V_e(P); \\text{SEP}]",
    contribution: "极大地简化了视觉-语言预训练的架构，显著提升了计算效率，证明了通过深度的跨模态交互直接学习对齐是可行的，推动了后续VLM向更简洁、更高效的架构演进。"
  },

  // =================================================================================
  // 第二部分：视觉-语言预训练 (Vision-Language Pre-training)
  // =================================================================================
  {
    id: "radford2021clip",
    title: "Learning Transferable Visual Models From Natural Language Supervision",
    authors: "Radford et al.",
    year: 2021,
    method: "CLIP (Contrastive Language-Image Pre-training)",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "通过在海量（4亿）网络图文对上进行对比学习，将图像和文本映射到同一个多模态嵌入空间，实现了强大的零样本（Zero-shot）图像Metric Learning能力。",
    formula: "\\mathcal{L} = -\\sum_{(i,j) \\in \\text{positive pairs}} \\log \\frac{\\exp(\\text{sim}(v_i, t_j)/\\tau)}{\\sum_{k=1}^{N} \\exp(\\text{sim}(v_i, t_k)/\\tau)}",
    contribution: "开创了利用自然语言作为监督信号进行大规模视觉预训练的新范式，其强大的零样本迁移能力深刻影响了后续多模态大模型的设计，是连接视觉和语言的桥梁性工作。"
  },
  {
    id: "li2021albef",
    title: "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation",
    authors: "Li et al.",
    year: 2021,
    method: "ALBEF",
    category: "Core Models & Architectures",
    influences: ["radford2021clip", "devlin2019bert", "kim2021vilt"],
    core_idea: "提出“先对齐，后融合”的策略：首先通过图文对比损失对齐单模态特征，然后通过多模态编码器进行深度融合，并利用动量蒸馏从噪声数据中学习。",
    formula: "\\mathcal{L} = \\mathcal{L}_{\\text{ITC}} + \\mathcal{L}_{\\text{ITM}} + \\mathcal{L}_{\\text{MLM}}",
    contribution: "在处理含噪图文对方面取得了显著效果，其“对齐-融合”思想和动量蒸馏技术为后续处理大规模网络数据的VLP模型提供了重要借鉴。"
  },
  {
    id: "li2022blip",
    title: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation",
    authors: "Li et al.",
    year: 2022,
    method: "BLIP",
    category: "Core Models & Architectures",
    influences: ["li2021albef", "wang2022simvlm"],
    core_idea: "设计了一个能同时处理理解与生成任务的统一框架，并通过“字幕生成与过滤”（CapFilt）的数据自举策略，为网络图片生成高质量的合成字幕以提升训练效果。",
    formula: "\\text{Unified Encoder-Decoder Architecture}",
    contribution: "成功统一了VLP中的理解和生成两大范式，其CapFilt数据增强策略为解决图文数据噪声问题提供了一个非常有效且可扩展的方案。"
  },
  {
    id: "yu2022coca",
    title: "CoCa: Contrastive Captioners are Image-Text Foundation Models",
    authors: "Yu et al.",
    year: 2022,
    method: "CoCa (Contrastive Captioners)",
    category: "Core Models & Architectures",
    influences: ["radford2021clip", "li2021albef"],
    core_idea: "在一个统一的Transformer架构中，通过解耦的注意力机制同时优化对比学习损失和字幕生成损失，将两种主流VLP范式高效地结合起来。",
    formula: "\\mathcal{L} = \\mathcal{L}_{\\text{contrastive}} + \\lambda \\mathcal{L}_{\\text{captioning}}",
    contribution: "构建了一个强大的图像-文本基础模型，在零样本Metric Learning、检索和生成任务上均表现出色，展示了多任务联合优化的巨大潜力。"
  },
  {
    id: "li2023blip2",
    title: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models",
    authors: "Li et al.",
    year: 2023,
    method: "BLIP-2",
    category: "Core Models & Architectures",
    influences: ["li2022blip", "alayrac2022flamingo"],
    core_idea: "提出一种轻量级的连接模块 Q-Former，用于连接一个固定的图像编码器和一个固定的LLM，在极低的训练成本下，高效地将视觉信息“翻译”成LLM能理解的语言。",
    formula: "\\text{Output} = \\text{LLM}(\\text{Q-Former}(\\text{ImageFeatures}), \\text{Text})",
    contribution: "为如何高效利用现有的强大单模态模型来构建多模态系统提供了一个极其成功且有影响力的范例，引领了后续一系列基于LLM的多模态模型（如LLaVA）的发展。"
  },
  {
    id: "wang2022simvlm",
    title: "SimVLM: Simple Visual Language Model Pretraining with Weak Supervision",
    authors: "Wang et al.",
    year: 2022,
    method: "SimVLM",
    category: "Core Models & Architectures",
    influences: ["dosovitskiy2020vit"],
    core_idea: "证明了通过极简的预训练目标（前缀语言模型 PrefixLM）和超大规模的弱监督图文数据，即可训练出性能强大的VLM，无需复杂的物体检测器或多任务设计。",
    formula: "\\mathcal{L}_{\\text{PrefixLM}} = -\\sum \\log p(\\text{token}_i | \\text{Image}, \\text{prefix})",
    contribution: "倡导了“大规模+简化”的VLP研究方向，证明了数据和模型规模在多模态学习中同样可以“大力出奇迹”，影响了后续基础模型的设计理念。"
  },
  {
    id: "bao2022beit",
    title: "BEiT: BERT Pre-Training of Image Transformers",
    authors: "Bao et al.",
    year: 2022,
    method: "BEiT / Masked Image Modeling",
    category: "Core Models & Architectures",
    influences: ["devlin2019bert", "dosovitskiy2020vit", "mae"],
    core_idea: "将BERT的掩码预测思想成功迁移到视觉领域，提出掩码图像建模（MIM）：随机掩盖图像块，然后让模型预测被掩盖块的离散视觉词元。",
    formula: "\\mathcal{L}_{\\text{MIM}} = -\\sum_{x_i \\in \\mathcal{M}} \\log p(x_i | x_{\\neg \\mathcal{M}})",
    contribution: "开创了视觉SSL学习的“掩码-预测”新范式（与MAE并称双雄），其学习到的表示在下游任务中展现了强大的迁移能力，也为多模态中的掩码建模提供了基础。"
  },

  // =================================================================================
  // 第三部分：多模态大语言模型 (Multimodal Large Language Models)
  // =================================================================================
  {
    id: "alayrac2022flamingo",
    title: "Flamingo: a Visual Language Model for Few-Shot Learning",
    authors: "Alayrac et al.",
    year: 2022,
    method: "Flamingo",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "通过巧妙的 Perceiver Resampler 和门控交叉注意力层，将一个预训练视觉模型和一个LLM连接起来，使其具备强大的少样本（Few-shot）上下文学习能力。",
    formula: "\\text{GatedCrossAttention}",
    contribution: "展示了如何构建能够理解图文交错上下文的强大VLM，其在少样本学习上的卓越表现，为后续多模态对话和上下文理解模型设定了很高的基准。"
  },
  {
    id: "liu2024llava",
    title: "LLaVA: Visual Instruction Tuning",
    authors: "Liu et al.",
    year: 2024,
    method: "LLaVA",
    category: "Core Models & Architectures",
    influences: ["li2023blip2", "alayrac2022flamingo"],
    core_idea: "开创性地提出视觉指令微调（Visual Instruction Tuning），通过一个简单的线性投影层连接视觉编码器和LLM，并利用GPT-4生成高质量的图文指令数据进行微调。",
    formula: "\\mathcal{L} = -\\sum \\log p(\\text{Answer} | \\text{Image}, \\text{Instruction})",
    contribution: "LLaVA以其简洁有效的方法和开源精神，极大地推动了多模态指令微调领域的发展，催生了大量后续工作，是构建多模态对话助手的里程碑。"
  },
  {
    id: "zhu2023minigpt4",
    title: "MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models",
    authors: "Zhu et al.",
    year: 2023,
    method: "MiniGPT-4",
    category: "Core Models & Architectures",
    influences: ["li2023blip2"],
    core_idea: "在BLIP-2架构基础上，增加了一个第二阶段的微调过程，使用少量高质量图文对，以对话格式训练模型，从而对齐LLM的输出风格，使其生成更自然、更详细的描述。",
    formula: "\\text{Two-stage finetuning}",
    contribution: "揭示了在连接视觉模块和LLM后，进行额外的“对话风格对齐”微调的重要性，为提升生成式VLM的输出质量提供了重要实践经验。"
  },
  {
    id: "dai2023instructblip",
    title: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning",
    authors: "Dai et al.",
    year: 2023,
    method: "InstructBLIP",
    category: "Core Models & Architectures",
    influences: ["li2023blip2", "alayrac2022flamingo"],
    core_idea: "在BLIP-2的Q-Former架构基础上，收集了涵盖26个不同数据集的、多样化的指令数据，进行大规模指令微调，显著提升了模型在未见过任务上的零样本泛化能力。",
    formula: "\\text{Multi-task Instruction Tuning}",
    contribution: "证明了指令数据的多样性和规模对于构建通用VLM至关重要，其在多个基准测试上的SOTA性能，确立了大规模指令微调在VLM领域的标准地位。"
  },
  {
    id: "peng2023kosmos2",
    title: "Kosmos-2: Grounding Multimodal Large Language Models to the World",
    authors: "Peng et al.",
    year: 2023,
    method: "Kosmos-2",
    category: "Core Models & Architectures",
    influences: ["karpathy2015alignments"],
    core_idea: "致力于解决多模态模型的“接地”（Grounding）问题，通过在文本输出中引入特殊的边界框坐标标记 `<bbox>`，让模型学会将文本短语与图像中的具体区域对应起来。",
    formula: "A `<bbox>`cat`</bbox>` is on the `<bbox>`mat`</bbox>`.",
    contribution: "推动了多模态模型从“看懂”图像内容到能够“指明”具体位置的跨越，为更精细的图文理解和交互（如视觉问答、机器人指令）提供了基础。"
  },
  {
    id: "driess2023palme",
    title: "PaLM-E: An Embodied Multimodal Language Model",
    authors: "Driess et al.",
    year: 2023,
    method: "PaLM-E",
    category: "Core Models & Architectures",
    influences: ["alayrac2022flamingo"],
    core_idea: "构建了一个具身（Embodied）多模态语言模型，能接收图像、文本指令和机器人状态等多模态输入，并输出文本形式的机器人动作指令，成功将LLM的能力迁移到机器人控制领域。",
    formula: "\\text{Action} = \\text{PaLM-E}(\\text{Image}, \\text{Instruction}, \\text{RobotState})",
    contribution: "展示了构建通用智能体的巨大潜力，并证明了多模态具身数据的训练可以与纯语言任务相互促进，是多模态技术走向物理世界的重要一步。"
  },
  {
    id: "bai2023qwenvl",
    title: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond",
    authors: "Bai et al.",
    year: 2023,
    method: "Qwen-VL",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "一个功能全面的多模态模型，特别增强了细粒度视觉定位、图像中的文字识别（OCR）以及多图对话能力，在中英文和细粒度理解场景表现卓越。",
    formula: "\\text{Multi-task Pre-training & Finetuning}",
    contribution: "作为业界领先的开源多模态模型之一，Qwen-VL以其全面的能力和强大的性能，推动了多模态技术在更广泛和更复杂场景下的应用。"
  },

  // =================================================================================
  // 第四部分：融合、对齐与特定任务 (Fusion, Alignment & Specific Tasks)
  // =================================================================================
  {
    id: "snoek2005fusion",
    title: "Early versus Late Fusion in Semantic Video Analysis",
    authors: "Snoek et al.",
    year: 2005,
    method: "Early vs. Late Fusion",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "系统性地比较了两种基本的多模态融合策略：早期融合（在特征层合并）和晚期融合（在决策层合并）。",
    formula: "N/A",
    contribution: "这篇经典论文通过实验得出结论，晚期融合通常比早期融合更鲁棒，为多模态系统设计提供了重要的指导原则，至今仍被广泛引用。"
  },
  {
    id: "andrew2013dcca",
    title: "Deep Canonical Correlation Analysis",
    authors: "Andrew et al.",
    year: 2013,
    method: "DCCA (Deep CCA)",
    category: "Techniques & Strategies",
    influences: ["hotelling1936cca"],
    core_idea: "将经典的线性对齐方法CCA扩展到非线性领域，使用深度神经网络来学习多模态数据之间更复杂的非线性变换，以最大化变换后表示的相关性。",
    formula: "\\max \\text{corr}(g(X), h(Y))",
    contribution: "是多模态表示学习领域中一个重要的早期探索，为后续使用深度学习进行非线性模态对齐和共享空间学习的方法奠定了基础。"
  },
  {
    id: "xu2018pointfusion",
    title: "PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation",
    authors: "Xu et al.",
    year: 2018,
    method: "PointFusion",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "提出一种用于3D物体检测的深度融合方法，通过一个融合网络将图像CNN特征与LiDAR点云中每个点的几何特征进行关联，实现信息的有效互补。",
    formula: "\\text{FusionNet}(\\text{ImageFeatures}, \\text{PointFeatures})",
    contribution: "展示了在自动驾驶等领域，通过模型级的深度融合可以比传统方法更有效地利用多传感器信息，是多模态感知领域的一个代表性工作。"
  },
  {
    id: "badrinarayanan2015segnet",
    title: "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
    authors: "Badrinarayanan et al.",
    year: 2015,
    method: "SegNet",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一种用于图像语义分割的编码器-解码器网络，其特点是解码器利用编码器在池化时保存的位置索引进行上采样，从而高效且精确地恢复边界细节。",
    formula: "\\text{Upsampling with max-pooling indices}",
    contribution: "虽然是单模态工作，但其高效的编码器-解码器架构和信息传递机制对许多多模态生成和翻译任务的模型设计产生了影响。"
  },
  {
    id: "srivastava2012dbm",
    title: "Multimodal learning with deep boltzmann machines",
    authors: "Srivastava & Salakhutdinov",
    year: 2012,
    method: "Multimodal DBM",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "早期使用深度玻尔兹曼机（DBM）学习图像和文本的联合概率分布的代表性工作，能够从未标注数据中学习共享表示，并用于跨模态生成和检索。",
    formula: "P(v, h) = \\frac{1}{Z} e^{-E(v,h)}",
    contribution: "为后续使用深度生成模型进行多模态学习提供了灵感，是无监督多模态表示学习的早期重要探索。"
  },
  {
    id: "karpathy2015alignments",
    title: "Deep visual-semantic alignments for generating image descriptions",
    authors: "Karpathy & Li",
    year: 2015,
    method: "Visual-Semantic Alignment",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "开创性地提出一个能够将句子中的词语与图像中特定区域进行隐式对齐的图文生成模型，为后续的注意力机制在多模态领域的应用奠定了基础。",
    formula: "\\text{Multimodal Embedding for Alignment}",
    contribution: "这项工作是理解多模态交互从全局匹配走向局部细粒度对齐的关键一步，深刻影响了后续的图像字幕、视觉问答等任务。"
  },
  {
    id: "li2021filip",
    title: "FILIP: Fine-grained Interactive Language-Image Pre-training",
    authors: "Yao et al.",
    year: 2021,
    method: "FILIP",
    category: "Techniques & Strategies",
    influences: ["radford2021clip"],
    core_idea: "改进了CLIP的全局对比学习，提出一种跨模态后期交互机制，通过计算图像块和文本词元之间的细粒度相似度来定义对比损失，从而学习更精细的局部对齐。",
    formula: "\\mathcal{L} = -\\log \\frac{\\exp(\\max_{j} \\text{sim}(v_i, t_j)/\\tau)}{\\dots}",
    contribution: "推动了图文对比学习从宏观对齐走向微观对齐，其细粒度交互思想在需要精确定位的下游任务中表现更佳。"
  },
  {
    id: "hotelling1936cca",
    title: "Canonical Correlation Analysis",
    authors: "Hotelling",
    year: 1936,
    method: "CCA",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "经典的多元统计分析方法，旨在寻找两组变量之间的线性投影，使得投影后的变量相关性最大化。",
    formula: "\\max_{\\alpha, \\beta} \\text{corr}(\\alpha^T X, \\beta^T Y)",
    contribution: "作为一种经典的显式对齐方法，CCA及其非线性扩展（如DCCA）在多模态学习中被广泛用于将不同模态的特征投影到一个相关的共享子空间中。"
  },
  {
    id: "chen2023cogvlm",
    title: "CogVLM: Visual Expert for Pretrained Language Models",
    authors: "Wang et al.",
    year: 2023,
    method: "CogVLM",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "通过在LLM的注意力层和前馈网络中增加一个可学习的“视觉专家”模块，实现深度的图文融合，从而在保持语言能力的同时，显著提升模型的细粒度视觉理解能力。",
    formula: "\\text{Attention with Visual Expert}",
    contribution: "解决了通用VLM在处理需要深度视觉理解的任务时性能下降的问题，为如何在LLM中实现更深层次的跨模态融合提供了新的架构思路。"
  },

  // =================================================================================
  // 第五部分：数据与基准 (Data & Benchmarks)
  // =================================================================================
  {
    id: "schuhmann2022laion5b",
    title: "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models",
    authors: "Schuhmann et al.",
    year: 2022,
    method: "LAION-5B Dataset",
    category: "Ecosystem & Evaluation",
    influences: ["radford2021clip"],
    core_idea: "发布了一个包含超过50亿图文对的公开大规模数据集，通过抓取网页数据并使用CLIP模型进行筛选构建，旨在为社区提供可用于训练下一代多模态模型的开源资源。",
    formula: "5.85 \\times 10^9 \\text{ image-text pairs}",
    contribution: "极大地推动了多模态大模型研究的民主化，使得更多的研究者能够复现和探索类似CLIP的大规模预训练，是许多著名开源模型（如Stable Diffusion）的基石。"
  },
  {
    id: "krishna2016genome",
    title: "Visual Genome: Connecting Language and Vision using Crowdsourced Dense Image Annotations",
    authors: "Krishna et al.",
    year: 2016,
    method: "Visual Genome Dataset",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一个大规模、结构化的图像概念数据集，为每张图片提供了极其丰富的标注，包括物体、属性、关系、区域描述和问答对，将图像内容表示为一个场景图。",
    formula: "\\text{Scene Graphs}",
    contribution: "为研究更深层次的图像理解、常识推理和视觉-语言对齐提供了宝贵的数据资源，对许多需要细粒度理解的多模态研究产生了深远影响。"
  },
  {
    id: "gadre2023datacomp",
    title: "DataComp: In Search of the Next Generation of Multimodal Datasets",
    authors: "Gadre et al.",
    year: 2023,
    method: "DataComp Benchmark",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一个用于系统性地研究和构建多模态数据集的基准框架，通过大量对比实验研究数据源、筛选策略和模型规模等变量的影响。",
    formula: "\\text{Systematic Data Filtering Comparison}",
    contribution: "为未来大规模数据集的构建提供了科学的指导，其核心发现——“筛选策略比数据源更重要”——深刻影响了后续的数据处理流程。"
  },
  {
    id: "sharma2018conceptual",
    title: "Conceptual Captions: A Cleaned, Hypernymed, Image Alt-Text Dataset for Automatic Image Captioning",
    authors: "Sharma et al.",
    year: 2018,
    method: "Conceptual Captions (CC) Dataset",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一个大规模（超300万）、相对高质量的图文字幕数据集，通过复杂的自动化流程对网络图片的alt-text进行清洗和规范化，使其更通用、更具描述性。",
    formula: "3.3M \\text{ cleaned image-caption pairs}",
    contribution: "作为早期高质量的大规模图文数据集，Conceptual Captions成为了许多经典视觉-语言模型（如ViLT, ALBEF）预训练的重要基石。"
  },
  {
    id: "srinivasan2021wit",
    title: "WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning",
    authors: "Srinivasan et al.",
    year: 2021,
    method: "WIT Dataset",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一个从维基百科提取的大规模、多语言图文数据集，包含超过3700万图文对，覆盖108种语言，数据质量高且富含知识性。",
    formula: "37.6M \\text{ multilingual pairs}",
    contribution: "极大地促进了多语言和跨语言多模态学习的研究，为构建能够理解多种文化的VLM提供了宝贵的数据资源。"
  },
  {
    id: "desai2021redcaps",
    title: "RedCaps: Web-curated Image-text Data Created by the People, for the People",
    authors: "Desai et al.",
    year: 2021,
    method: "RedCaps Dataset",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "一个从社交新闻网站Reddit收集的大规模图文数据集，数据更接近人类在真实社交场景中使用的图文内容，具有更强的“意图性”和“上下文”。",
    formula: "12M \\text{ pairs from 350 subreddits}",
    contribution: "为研究社交媒体中的多模态内容理解提供了一个独特的、更具社会性的视角，有助于模型学习更生活化、更多样的图文关系。"
  },
  {
    id: "zhang2021vinvl",
    title: "VinVL: Revisiting Visual Representations in Vision-Language Models",
    authors: "Zhang et al.",
    year: 2021,
    method: "VinVL",
    category: "Core Models & Architectures",
    influences: ["li2020oscar", "chen2020uniter"],
    core_idea: "核心论点是提升VLM性能的关键在于提升视觉表示的质量。通过使用一个在更大物体检测数据集上预训练的、更强大的物体检测器来提取视觉特征，显著提升了下游任务性能。",
    formula: "\\text{Performance} \\propto f(\\text{Quality}(\\text{VisualFeatures}))",
    contribution: "强调了高质量的、富含物体语义信息的视觉特征对于视觉-语言对齐的重要性，提醒社区在关注融合算法的同时，不能忽略输入特征的质量。"
  },
  {
    id: "chen2020uniter",
    title: "UNITER: UNiversal Image-TExt Representation Learning",
    authors: "Chen et al.",
    year: 2020,
    method: "UNITER",
    category: "Core Models & Architectures",
    influences: ["li2020oscar"],
    core_idea: "一个通用的视觉-语言表示学习模型，通过四个预训练任务（MLM, MRM, ITM, WRA）联合学习图像区域特征和文本词元的跨模态表示。",
    formula: "\\mathcal{L}_{\\text{MLM}} + \\mathcal{L}_{\\text{MRM}} + \\mathcal{L}_{\\text{ITM}} + \\mathcal{L}_{\\text{WRA}}",
    contribution: "以其全面的设计和强大的性能，成为了当时VLP领域的一个重要基准模型，为后续的多任务VLP研究提供了参考。"
  },
  {
    id: "li2020oscar",
    title: "OSCAR: Object-Semantics Aligned Pre-training for Vision-Language Tasks",
    authors: "Li et al.",
    year: 2020,
    method: "OSCAR",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "提出在VLP中应明确将图像中的物体标签作为连接图像和文本的“锚点”，通过输入“图像区域-物体标签-文本词元”三元组进行预训练，以促进对齐。",
    formula: "\\text{Input} = (\\text{ImageRegion}, \\text{ObjectTag}, \\text{WordToken})",
    contribution: "其引入物体语义作为显式对齐信号的思想，有效地解决了图文对齐中的歧义问题，显著提升了VLP模型的性能。"
  },
  {
    id: "li2018densefuse",
    title: "DenseFuse: A Fusion Approach to Infrared and Visible Images",
    authors: "Li & Wu",
    year: 2018,
    method: "DenseFuse",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "一种用于融合红外和可见光图像的无监督深度学习方法，通过采用稠密连接的编码器-解码器架构，有效保留多模态特征，生成信息丰富的融合图像。",
    formula: "\\text{Encoder-Decoder with Dense Connections}",
    contribution: "在多模态图像融合领域，为如何设计能有效保留各模态信息的网络结构提供了范例，特别是在无监督场景下展现了优越性。"
  },
  {
    id: "chen2020hgmf",
    title: "HGMF: Heterogeneous Graph-based Fusion for Multimodal Data with Incompleteness",
    authors: "Chen & Zhang",
    year: 2020,
    method: "HGMF",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "提出一种基于异构图的融合方法，专门用于处理存在数据缺失的多模态场景，通过构建超节点和异构图，有效利用所有可用的不完整数据。",
    formula: "\\text{GNN on Heterogeneous Graph}",
    contribution: "为解决现实世界中普遍存在的多模态数据缺失问题提供了一个强大而灵活的框架，在推荐系统、生物信息学等领域有重要应用价值。"
  },
  {
    id: "danapal2020yolorf",
    title: "YOLO-RF: Sensor Fusion of Camera and LiDAR Raw Data for Vehicle Detection",
    authors: "Danapal et al.",
    year: 2020,
    method: "YOLO-RF",
    category: "Techniques & Strategies",
    influences: [],
    core_idea: "探索了数据级的早期融合策略，将LiDAR点云数据转换为图像视角，并作为额外通道与RGB图像拼接，输入到改进的YOLOv3检测器中。",
    formula: "\\text{Input} = \\text{Concat}(\\text{RGB}, \\text{LiDARChannels})",
    contribution: "在自动驾驶领域，证明了在原始数据层面进行早期融合的有效性，能够让检测网络更早地利用多模态互补信息，提升检测性能。"
  },
  {
    id: "yin2024survey",
    title: "A Survey on Multimodal Large Language Models",
    authors: "Yin et al.",
    year: 2024,
    method: "Survey",
    category: "Ecosystem & Evaluation",
    influences: [],
    core_idea: "系统性地梳理了多模态大语言模型（MLLM）的关键技术、应用进展和面临的挑战（如幻觉、评测、鲁棒性）。",
    formula: "N/A",
    contribution: "作为一篇及时的综述，它为研究者快速了解MLLM领域的全貌、技术脉络和未来方向提供了全面的指南。"
  },
  {
    id: "liang2024foundations",
    title: "Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions",
    authors: "Liang et al.",
    year: 2024,
    method: "Survey",
    category: "Ecosystem & Evaluation",
    influences: ["baltrusaitis2018survey"],
    core_idea: "从更宏观和基础的视角，探讨了多模态学习的核心原理、挑战（表示异质性、对齐粒度、Techniques & Strategies等）和悬而未决的开放性问题。",
    formula: "N/A",
    contribution: "为领域内的研究提供了一个高层次的理论框架，通过提出深刻的开放性问题，启发和指导未来的研究方向。"
  },
  {
    id: "zhang2023internlm",
    title: "InternLM-XComposer: A Vision-Language Large Model for Advanced Text-Image Comprehension and Composition",
    authors: "Zhang et al.",
    year: 2023,
    method: "InternLM-XComposer",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "一款强大的图文多模态大模型，突出特点是其卓越的图文理解和创作（Composition）能力，能根据用户指令进行复杂的图文混合内容生成。",
    formula: "\\text{Multi-stage Training for Composition}",
    contribution: "代表了多模态大模型在内容创作领域的一个重要进展，展示了模型从简单的描述生成走向复杂的、结构化的多模态内容创作的潜力。"
  },
  {
    id: "young2024yi",
    title: "Yi: Open Foundation Models by 01.ai",
    authors: "Young et al.",
    year: 2024,
    method: "Yi-VL",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "基于强大的Yi系列开源语言模型构建的多模态模型，通过连接视觉编码器，使其具备出色的图文理解和对话能力，并以开源形式共享。",
    formula: "\\text{Based on Yi LLM}",
    contribution: "作为高性能的开源模型，Yi-VL进一步推动了多模态技术的普惠和应用落地，尤其是在中英文多模态场景中提供了强有力的基座。"
  },
  {
    id: "lu2022unifiedio",
    title: "Unified-IO: A Unified Model for Vision, Language, and Multimodal Tasks",
    authors: "Lu et al.",
    year: 2022,
    method: "Unified-IO",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "首个使用单一模型架构处理多种不同输入/输出格式任务的模型，将所有任务（图像生成、检测、分割、问答）都统一为“序列到序列”的格式。",
    formula: "\\text{Everything} \\rightarrow \\text{Sequence of Tokens}",
    contribution: "其极致的统一范式展示了构建通用人工智能模型的巨大潜力，证明了不同模态和任务可以在一个统一的框架下进行学习，并展现出惊人的零样本迁移能力。"
  },
  {
    id: "chen2023vast",
    title: "VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset",
    authors: "Chen et al.",
    year: 2023,
    method: "VAST",
    category: "Core Models & Architectures",
    influences: ["lu2022unifiedio"],
    core_idea: "一个旨在处理视觉、音频、字幕和文本四种模态的“全能”基础模型，能接收任意模态组合输入，并生成任意模态输出，展现强大的跨模态理解和生成能力。",
    formula: "\\text{Omni-modality Foundation Model}",
    contribution: "是迈向处理更广泛模态的通用人工智能的重要探索，为构建能够理解和生成更复杂多媒体内容的基础模型提供了范例。"
  },
  {
    id: "shen2023moe",
    title: "Scaling Vision-Language Models with Sparse Mixture of Experts",
    authors: "Shen et al.",
    year: 2023,
    method: "Sparse Mixture of Experts (MoE) for VLM",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "将稀疏混合专家（MoE）技术成功应用于VLM，通过将部分前馈网络替换为多个专家网络，在显著增加模型总参数量的同时，保持了计算成本基本不变。",
    formula: "y = \\sum_{i=1}^{n} G(x)_i E_i(x)",
    contribution: "为构建更大、更强的多模态模型提供了一条可行的、计算高效的技术路径，是解决大模型扩展性问题的关键技术之一。"
  },
  {
    id: "wang2023onepeace",
    title: "ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities",
    authors: "Wang et al.",
    year: 2023,
    method: "ONE-PEACE",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "旨在构建一个能处理任意多种模态的通用模型，采用模块化设计，包含一个共享的Transformer主干和多个模态适配器，通过联合预训练学习共享表示。",
    formula: "\\text{Shared Transformer} + \\text{Modality Adapters}",
    contribution: "其可扩展的架构为实现“无限模态”的通用人工智能提供了一个有前景的方向，推动了更通用、更灵活的多模态系统研究。"
  },
  {
    id: "chen2023xllm",
    title: "X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages",
    authors: "Chen et al.",
    year: 2023,
    method: "X-LLM",
    category: "Core Models & Architectures",
    influences: [],
    core_idea: "提出新颖的多模态学习框架，将所有非文本模态（如图像、语音）视为“外语”，通过一个共享的“翻译”模块将其输入都转换为LLM能理解的语言空间表示。",
    formula: "\\text{Modality} \\rightarrow \\text{Translate} \\rightarrow \\text{LLM}",
    contribution: "这种优雅的类比框架使得模型可以轻松扩展到多种模态，并展现了在多模态理解、生成和跨模态转换上的强大通用性。"
  },

];

const data = [...face, ...latent, ...multimodal]

// Export the papersData array for use in other files
if (typeof module !== "undefined" && module.exports) {
  module.exports = data;
} else if (typeof window !== "undefined") {
  window.papersData = data;
}