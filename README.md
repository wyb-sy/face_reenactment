including but not limited to face reenactment, talking head generation
(will be completed soon)

Comment in Chinese. I will update this document every week if not busy.

# Institutions
### NVIDIA, Netease Fuxi, Tencent AI, SenseTime, Samsung, Xiaobing, Baidu, Huawei

# Person (as far as i know)
#### [Liu Ziwei](https://liuziwei7.github.io/publications.html)
#### []

# Paper

### General keywords: audio-driven, stylegan, transformer, 3D, emotion, video, FOMM (including flow), landmark (not limited to keypoints), wav2lip, Nerf, 3DMM, diffusion, flow
### Other keyword: portrait editing, pose-controllable, edge map, depth map, 3D FOMM (indicating [this](https://arxiv.org/abs/2011.15126))

## 2023
### \[ICASSP] DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions
keywords: video, portrait editing
- [project page](https://deepbrainai-research.github.io/discohead/)
- [paper](https://arxiv.org/abs/2303.07697)
- [code](https://github.com/deepbrainai-research/discohead)

comment: 力大砖飞代表作。如果我不知道怎么解耦，那就制造解耦的条件。如果我不知道怎么融合信息，那就adain，如果我泛化性一般，那我就对特定人物做微调，对使用场景做限制。毕竟，华丽的方法论不如朴实的模型，朴实的模型不如海量的数据，某种意义上有一些工程实现上的启发（？）。不过有一说一视频展示效果还不错的，有一种米格25飞出3马赫的感觉。

### \[arxiv] Style Transfer for 2D Talking Head Animation
keywords: video, portrait editing
- [paper](https://arxiv.org/abs/2303.09799)
- [code](https://github.com/aioz-ai/AudioDrivenStyleTransfer) (not available for now)

comment: 在这个下面提一嘴，人脸相关的文章不放视频我个人感觉说服力很有限（不是针对这个文章，是针对所有不放视频的）

### \[CVPR] DPE: Disentanglement of Pose and Expression for General Video Portrait Editing
keywords: video, portrait editing
- [project page](https://carlyx.github.io/DPE/)
- [paper](https://arxiv.org/pdf/2301.06281)
- [code](https://github.com/Carlyx/DPE)

### \[AAAI] StyleTalk: One-shot Talking Head Generation with Controllable Speaking Styles
keywords: audio-driven, 3DMM, transformer, video, emotion
- [video](https://www.youtube.com/watch?v=mO2Tjcwr4u8)
- [paper](https://arxiv.org/abs/2301.01081)
- [code](https://github.com/FuxiVirtualHuman/styletalk)

comment: 由于只有下半张脸在动，全脸看起来略显僵硬，但音唇同步效果超出预料，transformer在多模态信息融合上还是王者。生成视频看起来不是很能保证speaker的id，可能是受制于3DMM，也可能是因为这不是网络主要关注的部分。几个discriminator的设计是有学习价值的。

### \[ICASSP] Memory-augmented Contrastive Learning for Talking Head Generation
keywords: audio-driven, landmark, video
- [video](https://www.youtube.com/watch?v=mO2Tjcwr4u8)
- [paper](https://arxiv.org/abs/2301.01081)
- [code](https://github.com/Yaxinzhao97/MACL)

comment: 还没细看，比较独特

### \[ICLR] GENEFACE: TOWARDS HIGH-FIDELITY AUDIO DRIVEN 3D TALKING FACE SYNTHESIS GENERALIZABLE TO OUT-OF-DOMAIN AUDIO
keywords: audio-driven, Nerf, landmark, video
- [project page](https://geneface.github.io/)
- [paper](https://openreview.net/forum?id=YfwMIDhPccD)
- [code](https://github.com/yerfor/GeneFace)

comment: HuBert的使用比较有借鉴意义，尤其是在工程上

### \[arxiv] Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation
keywords: audio-driven, diffusion, 
- [project page](https://mstypulkowski.github.io/diffusedheads/)
- [paper](https://arxiv.org/abs/2301.03396)

comment: 依据前两帧推后一帧，可能有误差累积，从展示视频来看效果比DiffTalk好一点，展示视频中没有长视频。

### \[arxiv] DiffTalk: Crafting Diffusion Models for Generalized Talking Head Synthesis
keywords: Diffusion, landmark, audio-driven
- [paper](https://arxiv.org/abs/2301.03786)
- [video](https://cloud.tsinghua.edu.cn/f/e13f5aad2f4c4f898ae7/)

- comment: 时序不稳定，不好说是diffusion的问题还是前面和landmark一起fusion的问题。使用diffusion应该也是个趋势。

## 2022
### \[Arxiv] SPACE: Speech-driven Portrait Animation with Controllable Expression
keywords: audio-driven, 3D FOMM, video, emotion
- [project page](https://deepimagination.cc/SPACEx/), 
- [paper](https://arxiv.org/pdf/2211.09809), 

comment: 只使用LSTM就能达到一个比较精准的面部landmark预测，比较有意思，而且生成效果非常好，会是很好的工业界应用。但文中提到的表情的引入效果不是很明显。

### \[FG] StyleMask: Disentangling the Style Space of StyleGAN2 for Neural ace Reenactment
keywords: stylegan
- [paper](https://arxiv.org/pdf/2209.13375)
- [code](https://github.com/StelaBou/StyleMask)

comment: 还没咋看, 好像挺独特

### \[] 
keywords:
project page, 
paper, 
code

### \[arxiv] MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation
keywords: flow, landmark (interesting), 
- [project page](https://meta-portrait.github.io), 
- [paper](https://arxiv.org/abs/2212.08062), 
- [code](https://github.com/Meta-Portrait/MetaPortrait)

comment: 效果很好，作者指出引入密集landmark对于细粒度的表情变化的捕捉比较有效，但作者使用的landmark提取器是未开源的。文章提取光流的部分使用了global vector来进行指导，比较有借鉴意义。文章提出使用meta-learning来加速微调，对于工业界应用比较有价值，但本人并不太理解为什么会那么有效。提出使用3D-Conv进行超分，idea很直观，有效性待验证。

### \[arxiv] High-fidelity Facial Avatar Reconstruction from Monocular Video with Generative Priors
keywords: Nerf, 3DMM, audio-driven, 3D-GAN
- [paper](https://arxiv.org/abs/2211.15064), 

### \[arxiv] StyleFaceV: Face Video Generation via Decomposing and Recomposing Pretrained StyleGAN3
keywords: stylegan, video, landmark
- [project page](http://haonanqiu.com/projects/StyleFaceV.html), 
- [paper](https://arxiv.org/abs/2208.07862), 
- [code](https://github.com/arthur-qiu/StyleFaceV)

### \[arxiv] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation
keywords: 3D, emotion, video, audio-driven, 3DMM
- [project page](https://sadtalker.github.io), 
- [paper](https://arxiv.org/abs/2211.12194), 
- [code](https://github.com/Winfredy/SadTalker)

### \[SIGGRAPH] VideoReTalking: Audio-based Lip Synchronization for Talking HeadVideo Editing In the Wild
keywords: audio-driven, video, wav2lip, transformer, emotion
- [project page](https://vinthony.github.io/video-retalking/), 
- [paper](https://arxiv.org/abs/2211.14758), 
- [code](https://github.com/vinthony/video-retalking/)

### \[SIGGRAPH] Masked Lip-Sync Prediction by Audio-Visual Contextual Exploitation in Transformers
keywords: Transformer, audio-driven, wav2lip, video
- [project page](https://hangz-nju-cuhk.github.io/projects/AV-CAT), 
- [paper](https://arxiv.org/abs/2212.04970), 

### \[CVPR] Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation
keywords: audio-driven, pose-controllable, video
- [project page](https://hangz-nju-cuhk.github.io/projects/PC-AVS), 
- [paper](https://arxiv.org/abs/2104.11116), 
- [code](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)

### \[CVPR] Depth-Aware Generative Adversarial Network for Talking Head Video Generation
keywords: FOMM, depth map
- [project page](https://harlanhong.github.io/publications/dagan.html), 
- [paper](https://arxiv.org/abs/2203.06605), 
- [code](https://github.com/harlanhong/CVPR2022-DaGAN)

### \[arxiv] One-Shot Face Reenactment on Megapixels
keywords: stylegan, 3DMM, landmark
- [paper](https://arxiv.org/pdf/2205.13368), 

### \[ECCV] StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN
keywords: stylegan, flow, audio-driven, 3DMM
- [project page](http://feiiyin.github.io/StyleHEAT/), 
- [paper](https://arxiv.org/abs/2203.04036), 
- [code](https://github.com/FeiiYin/StyleHEAT/)

### \[ECCV] Face2Faceρ: Real-Time High-Resolution One-Shot Face Reenactment
keywords: keypoint (interesting), 3DMM, flow
- [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730055.pdf), 
- [code](https://github.com/NetEase-GameAI/Face2FaceRHO)

### \[CVPR] Expressive Talking Head Generation with Granular Audio-Visual Control
keywords:
- [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Expressive_Talking_Head_Generation_With_Granular_Audio-Visual_Control_CVPR_2022_paper.pdf
)

### \[ACM MM] MegaPortraits: One-shot Megapixel Neural Head Avatars
- keywords: 3DMM, 3D FOMM, 
- [project page](https://samsunglabs.github.io/MegaPortraits/)
- [paper](https://arxiv.org/abs/2207.07621)
- code [None]

comment: 效果比较惊艳，是3D FOMM的改进版，把直接的三维表示换成了vector，但这样有点粘连的感觉，没有开源，比较遗憾。

### \[CVPR] Thin-Plate Spline Motion Model for Image Animation
keywords:
- [paper](https://arxiv.org/abs/2203.14367), 
- [code](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)

comment: FOMM系的改进产品，使用更复杂的变换代替仿射变换

### \[] Live Speech Portraits: Real-Time Photorealistic Talking-Head Animation
keywords:
project page, 
paper, 
code

## 2021
### \[ICCV] PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering
keywords:
- [project page](https://renyurui.github.io/PIRender_web/), 
- [paper](https://arxiv.org/abs/2109.08379), 
- [code](https://github.com/RenYurui/PIRender)

### \[CVPR] Audio-Driven Emotional Video Portraits
keywords: landmark, audio-driven, 3DMM, edge map，video
- [project page](https://jixinya.github.io/projects/evp/), 
- [paper](https://arxiv.org/abs/2104.07452), 
- [code](https://github.com/jixinya/EVP)

### \[ICCV] HeadGAN: One-shot Neural Head Synthesis and Editing
keywords: 3DMM, audio-driven, flow
- [project page](https://michaildoukas.github.io/HeadGAN/), 
- [paper](https://arxiv.org/abs/2012.08261), 

### \[ICCV] FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning
keywords: audio-driven, 3DMM
- [project page](https://personal.utdallas.edu/~xguo/), 
- [paper](https://arxiv.org/abs/2108.07938), 
- [code](https://github.com/zhangchenxu528/FACIAL)

### \[] One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing
keywords:
- [project page](https://nvlabs.github.io/face-vid2vid), 
- [paper](https://arxiv.org/abs/2011.15126), 
- [code](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) [unoffical]

### \[SIGGRAPH] Live Speech Portraits: Real-Time Photorealistic Talking-Head Animation
keywords:
- [paper](https://arxiv.org/abs/2109.10595), 
code

### \[] 
keywords:
project page, 
paper, 
code

### \[] 
keywords:
project page, 
paper, 
code

### \[IJCAI] Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion
keywords: audio-driven, FOMM
- [paper](https://www.ijcai.org/proceedings/2021/0152.pdf), 
- [code](https://github.com/wangsuzhen/Audio2Head)

## 2020
### \[ACM MM]\[Wav2Lip] A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild
keywords: video, wav2lip, audio-driven
- [project page](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/), 
- [paper](https://arxiv.org/abs/2008.10010), 
- [code](https://github.com/Rudrabha/Wav2Lip)

### \[ISMAR] Photorealistic Audio-driven Video Portraits
keywords: audio-driven, 3DMM, video
- [project page](https://richardt.name/publications/audio-dvp/), 
- [paper](http://miaowang.me/papers/advp_authors.pdf), 
- [code](https://github.com/xinwen-cs/AudioDVP)
