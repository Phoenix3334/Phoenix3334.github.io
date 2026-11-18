---
layout: post
title: "关于这个博客"
date: 2026-11-18 10:00:00 +0800
lang: zh-CN
summary: "记录 AI Infrastructure 和 LLM Systems 的学习、工程实践与可复现实验。"
tags: [AI-Infra, Notes]
---

这是我的技术笔记空间，主要记录 **LLM 推理系统、GPU/NPU 性能工程，以及训练与推理如何协同**。

比起简单收集论文和优化数字，我更希望每篇文章回答几个问题：

- **问题是什么？** 场景、输入负载以及性能或正确性约束。
- **证据是什么？** 基线、测试配置、Profiler、日志与可复现步骤。
- **为什么有效？** 从调度、通信、访存、算子或系统边界解释收益。
- **边界在哪里？** 区分 Microbenchmark、端到端收益，以及尚未验证的结论。

后续计划围绕 **SGLang Runtime、分布式推理、Speculative Decoding、CUDA/Triton、Ascend 和系统容量规划** 整理技术文章。

本篇是博客的第一篇说明文章，也用于验证 Markdown 发布流程。
