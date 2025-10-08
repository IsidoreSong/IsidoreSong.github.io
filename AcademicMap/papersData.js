// Formulas have been converted to valid LaTeX format
const data = [
  // 第一部分：基于分类的身份监督方法 (Classification-Based / Identity-Supervised Methods)
  
  // A. 奠基性工作：通过大规模分类学习特征
  {
    id: "deepid",
    title: "Deep Learning Face Representation from Predicting 10,000 Classes",
    authors: "Sun et al.",
    year: 2014,
    method: "DeepID",
    category: "分类",
    influences: [],
    core_idea: "将人脸识别视为一个大规模（10,000 类）分类任务，使用 Softmax 损失进行端到端训练，隐式学习判别性特征表示，无需额外的度量损失。",
    formula: "\\mathcal{L} = -\\log \\left( \\frac{e^{f(x_i)_{y_i}}}{\\sum_{j=1}^{C} e^{f(x_i)_j}} \\right)",
    contribution: "开创性地验证了深度学习在人脸识别上的巨大潜力，提出“通过分类任务隐式学习判别性特征”的范式，成为后续研究的基础。"
  },
  {
    id: "deepid2",
    title: "Deep Learning Face Representation by Joint Identification-Verification",
    authors: "Sun et al.",
    year: 2014,
    method: "DeepID2",
    category: "分类",
    influences: ["deepid"],
    core_idea: "在 DeepID 基础上，融合识别损失（Softmax）与验证损失（对比损失，Contrastive Loss），实现双重优化：类内聚合 + 类间分离。",
    formula: "\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2N} \\sum_{i=1}^{N} \\left[ y_i \\cdot d_i^2 + (1 - y_i) \\cdot \\max(0, m - d_i)^2 \\right]",
    contribution: "第一次将「识别」与「验证」双重任务联合训练，是早期将分类与度量学习结合的典范，显著提升了跨姿态与光照下的识别鲁棒性。"
  },
  {
    id: "deepid2plus",
    title: "Deeply learned face representations are sparse, selective, and robust",
    authors: "Sun et al.",
    year: 2015,
    method: "DeepID2+",
    category: "分类",
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
    category: "分类",
    influences: ["deepid"],
    core_idea: "在传统 Softmax 损失之外，引入一个额外的“中心损失”（Center Loss），显式约束同一类别的特征向量靠近其类中心，从而增强类内紧凑性，同时保留 Softmax 的类间分离能力。",
    formula: "\\mathcal{L}_{\\text{center}} = \\frac{1}{2} \\sum_{i=1}^{N} \\| x_i - c_{y_i} \\|^2",
    contribution: "首次显式建模类内紧凑性，证明仅靠 Softmax 无法有效控制类内方差，为后续度量学习与角度损失铺平道路。"
  },
  {
    id: "normface",
    title: "NormFace & L2-constrained Softmax Loss for Discriminative Face Verification",
    authors: "Wang et al.",
    year: 2017,
    method: "L2-Softmax",
    category: "分类",
    influences: ["centerloss"],
    core_idea: "提出将特征和类别权重同时进行 L2 归一化，将所有特征“投影”到单位超球面上。这使得优化目标从同时考虑特征的“模长+角度”简化为“仅优化角度”，分类决策完全由余弦相似度（即夹角）决定，从而消除了特征模长对分类的干扰。",
    formula: "\\mathcal{L}_{\\text{L2-Softmax}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos\\theta_{y_i}}}{\\sum_{j=1}^{C} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "为后续所有角度损失（ArcFace、CosFace）奠定了理论和几何基础，阐明了“特征归一化+度量学习”是提升人脸识别性能的关键路径，确立了“超球面嵌入”的标准范式。"
  },
  {
    id: "lsoftmax",
    title: "Large-Margin Softmax Loss for Convolutional Neural Networks",
    authors: "Liu et al.",
    year: 2016,
    method: "L-Softmax",
    category: "分类",
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
    category: "分类",
    influences: ["lsoftmax", "normface"],
    core_idea: "在 L-Softmax 的基础上，提出“乘性角度间隔”（multiplicative angular margin），将类别中心与样本特征之间的夹角放大 m 倍（m > 1），从而在超球面上强制形成更宽的分类边界，显著增强类间分离。",
    formula: "\\mathcal{L}_{\\text{SphereFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{s \\cdot \\cos(m \\theta_{y_i})}}{e^{s \\cdot \\cos(m \\theta_{y_i})} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "首次证明乘性角度间隔比加性间隔在几何上更具判别力，是首个在超球面上实现强分开的损失函数，为 ArcFace 和 CosFace 提供了直接启发。"
  },
  {
    id: "cosface",
    title: "CosFace: Large Margin Cosine Loss for Deep Face Recognition",
    authors: "Wang et al.",
    year: 2018,
    method: "CosFace",
    category: "分类",
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
    category: "分类",
    influences: ["cosface", "sphereface"],
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
    category: "分类",
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
    category: "分类",
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
    category: "分类",
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
    category: "分类",
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
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "不是单纯拉近正类中心或推开负类，而是主动利用被错误分类样本的特征向量与它被误分到的类别中心之间的方向信息，来指导如何更精准地调整其与真实类中心的距离。",
    formula: "\\mathcal{L}_{\\text{MV}} = \\mathcal{L}_{\\text{ArcFace}} + \\lambda \\cdot \\| \\frac{v_i}{\\|v_i\\|} \\cdot (c_{y_i} - x_i) \\|",
    contribution: "将“判别性学习”从被动边界扩张，升级为主动错误引导修正，是首个基于错误样本反馈机制的损失函数，对噪声标签和边界混淆样本有显著抑制作用。"
  },
  {
    id: "scfarcface",
    title: "Spherical Confidence Learning for Face Recognition",
    authors: "Chen et al.",
    year: 2023,
    method: "SCF-ArcFace",
    category: "分类",
    influences: ["arcface"],
    core_idea: "将模型对样本分类的“置信度”（即当前预测概率）作为调整角度间隔的信号：对于低置信度样本（即模型不确定的难样本），施加更大间隔以强化学习；对于高置信度样本减小约束，避免过度拟合。",
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
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "人类学习遵循“由易到难”的曲线。CurricularFace 模仿这一过程，根据每个样本当前的分类难度（由其与目标类中心的余弦相似度 cosθ_i 与平均相似度 m̄ 的差值衡量），动态调整其在损失函数中的权重，使模型先学习简单样本，再逐步接触难样本。",
    formula: "\\mathcal{L}_{\\text{CurricularFace}} = -\\frac{1}{N} \\sum_{i=1}^{N} w_i \\cdot \\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m)}}{e^{s \\cdot \\cos(\\theta_{y_i} + m)} + \\sum_{j \\ne y_i} e^{s \\cdot \\cos\\theta_j}}",
    contribution: "提出“难度感知的课程学习”机制，避免模型早期被困难样本主导而陷入局部最优。该方法显著降低训练震荡，提高收敛稳定性，在 MegaFace、LFW 等基准上刷新当时 SOTA。"
  },
  {
    id: "adacos",
    title: "AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations",
    authors: "Zhong et al.",
    year: 2019,
    method: "AdaCos",
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "Vectorized 条件下的角度损失（如 ArcFace）在训练后期容易出现“梯度饱和”或“梯度爆炸”，尤其在难样本附近。P2SGrad 并不改变损失函数本身，而是重新设计梯度传递过程，构造更平滑、可控的更新方向。",
    formula: "\\nabla_{\\theta_i} \\mathcal{L} = \\gamma_i \\cdot \\nabla_{\\theta_i} \\mathcal{L}_{\\text{ArcFace}}",
    contribution: "将优化器从“损失驱动”升级为“梯度流驱动”，解决了角度损失在后期训练中收敛不稳定的问题，加速模型达到泛化最优解，已在 PyTorch 社区开源并被广泛采用。"
  },
  
  // 特定场景解决方案
  {
    id: "qgface",
    title: "QGFace: Quality-Guided Joint Training For Mixed-Quality Face Recognition",
    authors: "Li et al.",
    year: 2022,
    method: "QGFace",
    category: "分类",
    influences: ["arcface"],
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
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "与其分别学习 HR 与 LR 的特征，不如构建一个“耦合特征空间”，使得来自不同分辨率的样本能直接比较。TCN 学习一个跨分辨率的联合嵌入函数。",
    formula: "\\text{同人：}\\| f_{\\text{HR}}(x) - f_{\\text{LR}}(x') \\|^2 \\leq \\delta; \\text{不同人：}\\| f_{\\text{HR}}(x) - f_{\\text{LR}}(y) \\|^2 \\geq \\delta + m",
    contribution: "提出“跨域耦合嵌入”范式，将问题从“HR→LR 映射”转变为“统一度量学习”，实现真正的跨分辨率人脸识别，成为工业级视频监控系统的基础组件。"
  },
  {
    id: "ideanet",
    title: "IDEA-Net: An Implicit Identity-Extended Data Augmentation for Low-Resolution Face Representation Learning",
    authors: "Chen et al.",
    year: 2022,
    method: "IDEA-Net",
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "传统方法处理单张图像，而集合人脸识别（如监控视频中的多帧人脸）需建模图像集合内部的多样性与集合间的分离性。DDCFM 同时学习“判别性表示”和“集合间距离度量”。",
    formula: "d(\\mathcal{S}_i, \\mathcal{S}_j) = \\| \\phi(\\mathcal{S}_i) - \\phi(\\mathcal{S}_j) \\|",
    contribution: "首次建立“集合-集合而非单样本-单样本”的识别框架，适用于视频监控、空中人脸识别等实际应用，为序列式人脸识别提供理论基础。"
  },
  {
    id: "headposesoftmax",
    title: "HeadPose-Softmax: Head Pose Adaptive Curriculum Learning Loss for Deep Face Recognition",
    authors: "Zhou et al.",
    year: 2021,
    method: "HeadPose-Softmax",
    category: "分类",
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
    category: "分类",
    influences: ["arcface"],
    core_idea: "在超分辨率任务中，传统方法仅恢复像素细节，忽略身份一致性。该文提出“超身份损失”：利用目标身份的多张高分辨率图像，构建一个身份原型中心 c_i，并要求生成的高清图像特征必须靠近该中心，从而保证身份不变性。",
    formula: "\\mathcal{L}_{\\text{super-id}} = \\| f_{\\text{SR}}(x_{\\text{LR}}) - c_i \\|^2",
    contribution: "将人脸识别中的“样本-类别关系”作为生成模型的约束工具，使超分辨率不仅是“变清晰”，更是“变对人”，在公安图像恢复、历史人物重建等领域具有重要价值。"
  },
  
  // 第二部分：基于度量学习的对比方法 (Metric Learning-Based / Contrastive Methods)
  
  // A. 监督对比学习 (Supervised Contrastive Learning)
  {
    id: "contrastiveloss",
    title: "Dimensionality Reduction by Learning an Invariant Mapping",
    authors: "Hadsell et al.",
    year: 2006,
    method: "Contrastive Loss",
    category: "度量",
    influences: [],
    core_idea: "首次提出“对比损失”（Contrastive Loss）框架，为深度度量学习奠定基石。其目标不是分类，而是学习一个映射函数 f，使得相同类别的样本在特征空间中靠近，不同类别的样本被推开至少一个安全边界。",
    formula: "\\mathcal{L}_{\\text{contrastive}} = \\frac{1}{2N} \\sum_{i,j} \\left[ y_{ij} \\cdot d_{ij}^2 + (1 - y_{ij}) \\cdot \\max(0, m - d_{ij})^2 \\right]",
    contribution: "首次将“相对距离约束”引入神经网络训练，实现非参数化度量学习，在MNIST、人脸识别早期实验中验证了有效性，成为后续所有对比/三元组损失的直接灵感来源。"
  },
  {
    id: "facenet",
    title: "FaceNet: A Unified Embedding for Face Recognition and Clustering",
    authors: "Schroff et al.",
    year: 2015,
    method: "Triplet Loss",
    category: "度量",
    influences: ["contrastiveloss"],
    core_idea: "提出“三元组损失”（Triplet Loss），以其强大的判别能力成为度量学习的里程碑。其核心是：通过构建“锚点-正样本-负样本”三元组，强制锚点到正样本的距离比到负样本的距离小一个安全间隔，从而实现端到端的特征嵌入优化。",
    formula: "\\mathcal{L}_{\\text{triplet}} = \\frac{1}{N} \\sum_{i=1}^{N} \\max \\left( \\|f(x_a^i) - f(x_p^i)\\|^2 - \\|f(x_a^i) - f(x_n^i)\\|^2 + m, 0 \\right)",
    contribution: "首次在大规模人脸数据集（LFW, YCC）上实现近人类性能的识别准确率（99.63%），并证明“固定维度嵌入”可同时完成识别、聚类、验证三大任务，成为工业标准，直至今日仍在许多系统中使用。"
  },
  {
    id: "lifted",
    title: "Deep Metric Learning via Lifted Structured Feature Embedding",
    authors: "Sohn",
    year: 2016,
    method: "Lifted Structured Loss",
    category: "度量",
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
    method: "SupCon",
    category: "度量",
    influences: ["simclr"],
    core_idea: "将对比学习从无监督扩展到全监督场景，突破传统对比学习仅以“样本对”为单位的限制——将同一类别的所有样本视为正样本对，不同类别样本为负样本，从而大幅提升正样本数量，显著提升类内紧凑性。",
    formula: "\\mathcal{L}_{\\text{supCon}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j \\in \\mathcal{P}_i} \\log \\frac{\\exp(s \\cdot \\cos(\\theta_{ij})) / \\tau}{\\sum_{k \\ne i} \\exp(s \\cdot \\cos(\\theta_{ik})) / \\tau}",
    contribution: "超越传统对比学习中“只能有一个正样本”的限制，通过类内全连接构建密集正样本信号，使得特征分布呈现极强的类内聚集和类间分离，在ImageNet等大规模分类任务中显著优于 Softmax，是现代自监督/半监督学习的基石之一。"
  },
  {
    id: "circleloss",
    title: "Circle Loss: A Unified Perspective of Pair Similarity Optimization",
    authors: "Sun et al.",
    year: 2020,
    method: "Circle Loss",
    category: "度量",
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
    category: "度量",
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
    category: "度量",
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
    category: "度量",
    influences: ["contrastiveloss"],
    core_idea: "模型训练时不依赖身份分类标签，而是以成对图像为单位构建样本：若两张图像是同一人（即使均为低清），则标签为正；若为不同人，则为负。通过对比损失优化特征空间，使低分辨率特征仍能承载身份判别信息。",
    formula: "\\mathcal{L}_{\\text{siamese}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\cdot \\log \\sigma(d_i) + (1 - y_i) \\cdot \\log (1 - \\sigma(d_i)) \\right]",
    contribution: "首次在低分辨率人脸识别中放弃超分辨率重建路径，转而直接在特征空间建立“匹配→鉴别”范式，大幅降低计算开销，适用于嵌入式设备与实时监控，在公安、交通等真实场景中被广泛借鉴。"
  },
  {
    id: "crossresolution",
    title: "Cross-resolution Learning for Face Recognition",
    authors: "Li et al.",
    year: 2019,
    method: "Cross-resolution Learning",
    category: "度量",
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
    category: "度量",
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
    category: "度量",
    influences: ["supcon"],
    core_idea: "在口罩遮挡下，传统方法因局部特征丢失而性能骤降。FocusFace 提出多任务对比学习框架：同时优化两个目标：全局身份对比：拉近完整脸、部分遮挡脸之间特征（正对）；局部区域聚焦：强制模型关注非遮挡区域（如眼睛、额头），通过注意力机制加权这些区域的特征贡献。",
    formula: "\\mathcal{L}_{\\text{global}} = \\mathcal{L}_{\\text{supCon}}; \\mathcal{L}_{\\text{focus}} = \\sum_{p \\in \\text{unmasked}} w_p \\cdot \\left( \\|f_p^a - f_p^p\\|^2 - \\|f_p^a - f_p^n\\|^2 \\right)",
    contribution: "首次将“遮挡鲁棒性”建模为注意力引导的对比学习任务，而非简单数据增强或特征掩码，在 RGB-D 和真实口罩数据集（WFR-Masked）上达到当时 SOTA，是疫情防控期间人脸识别技术的核心突破之一。"
  },
  
  // B. 自监督对比学习 (Self-Supervised Contrastive Learning)
  {
    id: "simclr",
    title: "A Simple Framework for Contrastive Learning of Visual Representations",
    authors: "Chen et al.",
    year: 2020,
    method: "SimCLR",
    category: "自监督",
    influences: ["nce", "instadisc"],
    core_idea: "SimCLR 提出了一种无需监督标签、仅依赖数据增强的对比学习框架。其核心思想是：对同一张图像通过随机增强（如裁剪、颜色扭曲、高斯模糊等）生成两个视图，将其作为正样本对，而批次中其他所有图像的增强视图则作为负样本。模型通过对比损失函数最大化正样本对的互信息，同时最小化与负样本的相似度。",
    formula: "\\mathcal{L}_{\\text{SimCLR}} = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k=1}^{2N} \\mathbb{1}_{[k \\neq i]} \\exp(\\text{sim}(z_i, z_k) / \\tau)}",
    contribution: "系统地验证了增强策略、投影头和批次大小对性能的决定性影响，并在多个下游任务上显著超越了当时的监督与自监督方法。"
  },
  {
    id: "simclrv2",
    title: "Big Self-Supervised Models are Strong Semi-Supervised Learners",
    authors: "Chen et al.",
    year: 2020,
    method: "SimCLR v2",
    category: "自监督",
    influences: ["simclr"],
    core_idea: "SimCLR v2 在 SimCLR 基础上进行了三重升级：(1) 使用更大的 ResNet 模型；(2) 引入更深更复杂的投影头；(3) 首次系统研究了自监督预训练在半监督场景下的迁移能力。",
    formula: "\\text{沿用 InfoNCE，但通过更强的表示能力和更精细的训练策略显著提升了表示质量}",
    contribution: "为后续“大模型+无标签数据”的范式铺平了道路。"
  },
  {
    id: "moco",
    title: "Momentum Contrast for Unsupervised Visual Representation Learning",
    authors: "He et al.",
    year: 2020,
    method: "MoCo",
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
    influences: ["byol"],
    core_idea: "SimSiam 进一步简化了 BYOL，移除了动量编码器。它使用一个对称孪生网络，仅在其中一个分支上施加停止梯度（stop-gradient）操作，切断反向传播。",
    formula: "\\mathcal{L}_{\\text{SimSiam}} = -\\frac{1}{2} \\left( \\frac{p(z_1)}{\\|p(z_1)\\|_2} \\cdot \\frac{\\text{stop\\_grad}(z_2)}{\\|\\text{stop\\_grad}(z_2)\\|_2} + \\frac{p(z_2)}{\\|p(z_2)\\|_2} \\cdot \\frac{\\text{stop\\_grad}(z_1)}{\\|\\text{stop\\_grad}(z_1)\\|_2} \\right)",
    contribution: "为自监督学习提供了最轻量、最直观的实现范式，证明对抗坍塌的关键是“预测目标与自身输出分离”的结构机制。"
  },
  {
    id: "dino",
    title: "Emerging Properties in Self-Supervised Vision Transformers",
    authors: "Caron et al.",
    year: 2021,
    method: "DINO",
    category: "自监督",
    influences: ["byol", "swav"],
    core_idea: "DINO 是首个将自监督学习成功应用于 Vision Transformer（ViT）的工作，并引入了自蒸馏框架。",
    formula: "\\mathcal{L}_{\\text{DINO}} = -\\frac{1}{2} \\sum_{i} \\sum_{j} q_j^{(i)} \\log p_j^{(i)} + \\text{reversed term}",
    contribution: "其自蒸馏过程使得模型自发地学习到语义一致的聚类结构，标志着自监督学习从“判别任务”转向“生成语义结构”的拐点。"
  },
  
  // 聚类与特征空间约束
  {
    id: "invaspread",
    title: "Unsupervised Embedding Learning via Invariant and Spreading Instance Feature",
    authors: "Ye et al.",
    year: 2019,
    method: "InvaSpread",
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
    influences: [],
    core_idea: "这是早期代表性的代理任务（proxy task）类自监督方法。给定一张图像，将其随机旋转 0°、90°、180° 或 270°，然后让模型预测其旋转角度。",
    formula: "\\mathcal{L}_{\\text{Rot}} = -\\sum_{c=1}^{4} y_c \\log p_c",
    contribution: "证明了通过设计一个合理、易于优化的自监督代理任务，同样可以提取出具有判别力的特征，为后续“自监督即代理任务”的范式铺平了道路。"
  },
  
  // 理论基石与跨领域启发
  {
    id: "nce",
    title: "Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics",
    authors: "Gutmann & Hyvärinen",
    year: 2010,
    method: "NCE",
    category: "自监督",
    influences: [],
    core_idea: "NCE 作为现代对比学习的理论基石，最早用于从无归一化（unnormalized）的概率模型中估计参数。其基本思想是将“概率密度估计”转化为一个二元分类问题。",
    formula: "\\mathcal{L}_{\\text{NCE}} = -\\mathbb{E}_{x \\sim p_d} \\left[ \\log \\frac{p_d(x)}{p_d(x) + k p_n(x)} \\right] - \\mathbb{E}_{x \\sim p_n} \\left[ \\log \\frac{k p_n(x)}{p_d(x) + k p_n(x)} \\right]",
    contribution: "将自监督学习从“启发式技巧”提升为“可证明的统计推断框架”，为 InfoNCE、SimCLR、MoCo 的损失函数提供了理论解释。"
  },
  {
    id: "word2vec",
    title: "Distributed Representations of Words and Phrases and their Compositionality",
    authors: "Mikolov et al.",
    year: 2013,
    method: "Word2Vec",
    category: "自监督",
    influences: [],
    core_idea: "虽然 Word2Vec 并非视觉方法，但其提出的“上下文预测”机制是现代自监督学习的鼻祖。它通过局部共现的统计模式在向量空间中编码语义关系。",
    formula: "\\mathcal{L}_{\\text{Skip-gram}} = \\sum_{(w, c) \\in D} \\log \\sigma(v_c^\\top v_w)",
    contribution: "揭示了“关系即结构”的思想，被直接迁移到对比学习中，是从“统计学习”迈向“几何表征学习”的关键一步。"
  },
  {
    id: "simcse",
    title: "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    authors: "Gao et al.",
    year: 2021,
    method: "SimCSE",
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
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
    category: "分析",
    influences: [],
    core_idea: "这篇工作系统比较了两类特征学习范式：一类以 Softmax 分类（如 ImageNet 监督分类）为损失函数，另一类以度量学习（如 Triplet Loss、Contrastive Loss）为优化目标。",
    formula: "N/A",
    contribution: "这一结论颠覆了当时“度量学习更优”的主流预期，并推动了后续基于 Softmax 的改进方法成为人脸识别的标准方案。它提醒我们：监督信号的强度和损失函数的结构设计同样重要，不能单纯依赖“对比”的形式化机制。"
  },
  {
    id: "noise",
    title: "The Devil of Face Recognition Is in the Noise",
    authors: "Wang et al.",
    year: 2018,
    method: "Data Analysis",
    category: "分析",
    influences: [],
    core_idea: "这是一篇极具现实意义的数据分析论文，作者分析了超过 10 个主流人脸识别数据集（包括 MS-Celeb-1M、WebFace、LFW），发现其中存在大量标签噪声。",
    formula: "N/A",
    contribution: "彻底改变了研究社区对数据质量的认知，强调“数据比模型更重要”。其结论推动了多个数据清洗工具、噪声鲁棒训练策略的发展，是自监督学习从“理想实验”走向“真实世界”的重要警钟。"
  },
  
  // 新增的论文
  {
    id: "otclip",
    title: "OT-CLIP: Understanding and Generalizing CLIP via Optimal Transport",
    authors: "Long et al.",
    year: 2022,
    method: "Optimal Transport",
    category: "自监督",
    influences: ["clip"],
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
    category: "自监督",
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
    category: "自监督",
    influences: ["byol", "simsiam"],
    core_idea: "挑战了主流自监督学习范式。它认为，无论是需要大量负样本的对比学习，还是需要动量编码器和预测头的非对比学习，都过于复杂。BaseAlign 提出一种极简的非对比学习框架：只需将当前模型对图像的表示，与该图像在先前训练阶段的“旧”表示进行对齐即可。",
    formula: "\\mathcal{L} = \\left\\| \\frac{z_i}{\\|z_i\\|_2} - \\frac{b_i}{\\|b_i\\|_2} \\right\\|_2^2",
    contribution: "提供了迄今最简单的非对比学习框架之一，揭示了避免模型坍塌的新机制，实现了高效的实现。"
  },
  {
    id: "pq",
    title: "Patch-Level Contrastive Learning via Positional Query for Visual Pretraining",
    authors: "Zhang et al.",
    year: 2023,
    method: "Positional Query",
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
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
    category: "自监督",
    influences: ["mae"],
    core_idea: "MaskFeat 是一种基于特征预测的自监督学习方法，不是重建像素，而是重建人类设计的特征（如 HOG）。这种方法更高效，在视频理解中实现突破。",
    formula: "\\mathcal{L}_{\\text{MaskFeat}} = \\sum_{m \\in M} \\| \\hat{f}_m - f_m \\|_2^2",
    contribution: "在 Kinetics-400 上以 MViT-L 达到 86.7%，远超同期方法，且计算效率极高。"
  },
  {
    id: "mae",
    title: "Masked Autoencoders Are Scalable Vision Learners",
    authors: "He et al.",
    year: 2021,
    method: "MAE",
    category: "自监督",
    influences: ["bert"],
    core_idea: "MAE 是视觉领域 BERT 的图灵级实现，通过遮蔽公式与不对称编解码结构，实现了高效且强大可扩展的自监督视觉预训练。",
    formula: "\\mathcal{L}_{\\text{MAE}} = \\frac{1}{N} \\sum_{i=1}^N \\| \\hat{x}_i - x_i \\|_2^2",
    contribution: "能训练 ViT-Huge，在 ImageNet-1K 上达到 87.8% 线性评估准确率，超越监督预训练。"
  }
];

// Export the papersData array for use in other files
if (typeof module !== "undefined" && module.exports) {
  module.exports = data;
} else if (typeof window !== "undefined") {
  window.papersData = data;
}