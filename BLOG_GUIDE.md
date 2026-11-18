# 博客写作与发布指南

博客地址：<https://phoenix3334.github.io/blog/>

这里使用 **GitHub Pages + Jekyll**。以后直接修改 Markdown 即可，不需要写 HTML、部署服务器或手工更新目录。

## 在 GitHub 网页发布一篇文章

1. 打开 [仓库的 _posts 目录](https://github.com/Phoenix3334/Phoenix3334.github.io/tree/main/_posts)。
2. 点击 **Add file → Create new file**（或者打开仓库右上角快捷创建文件）。
3. 文件名使用：\`YYYY-MM-DD-your-post-slug.md\`，例如 \`2026-11-18-sglang-dp-health.md\`。建议英文小写、短横线分隔。
4. 复制下面的模板，替换标题、日期和正文，点击 **Commit changes**。如果你不想马上发布，先创建在 \`_drafts/\`，等写好再移入 \`_posts/\`。

\`\`\`markdown
---
layout: post
title: "我的第一篇技术文章"
date: 2026-11-18 20:00:00 +0800
lang: zh-CN
summary: "一句话介绍这篇文章"
tags: [SGLang, Inference]
---

## 背景

这里写问题与测试场景。

## 分析

这里写原理、数据、代码或流程图。

## 结论

这里写结果、限制与下一步。
\`\`\`

Jekyll 会自动从 \`_posts/\` 生成文章页面，并将新文章显示在 \`/blog/\`，日期新的排在前面。文章网址采用：
\`/blog/YYYY/MM/DD/your-post-slug/\`。

## 编辑、撤稿、草稿

- 编辑文章：直接在 GitHub 的 \`_posts/*.md\` 文件点击铅笔，保存提交即可更新。
- 先保存草稿：放在 \`_drafts/\`；草稿默认不会公开生成。
- 暂时隐藏文章：将 front matter 改成 \`published: false\`。
- 删除文章：删除相应 Markdown 文件即可。
- 图片：上传到 \`assets/posts/\`，正文使用 \`![说明](/assets/posts/your-image.png)\`。
- 数学公式：现阶段没有额外的 KaTeX/MathJax 加载，涉及复杂公式时需要再增加支持。
- 代码块：直接使用三个反引号，带语言名称，例如 \`python\` 或 \`bash\`。
- 内容涉及工作项目时：只发布可公开的内容，避免内部日志、模型路径、生产数据或保密信息。

## 文件结构

\`\`\`text
index.html                 # 原有双语个人主页（保持）
_config.yml                # Jekyll 配置
_layouts/blog.html         # 博客框架
_layouts/post.html         # 单篇文章模板
blog/index.html            # 文章目录
blog/assets/blog.css       # 样式
_posts/*.md                # 已发布文章
_drafts/*.md               # 未发布草稿
\`\`\`

## 如果 GitHub Pages 没有自动发布

检查仓库 **Settings → Pages**，发布源需要是 **Deploy from a branch**，分支选 \`main\`、目录选 \`/(root)\`。GitHub Pages 在分支发布模式下默认使用 Jekyll；仓库根目录的 \`.nojekyll\` 已删除。

在仓库 **Actions** 查看最新 Pages 构建结果。Jekyll 更新后通常需要等部署完成才能访问新文章。如发现旧链接或报错，先检查文章头部的 \`---\` 配置和构建日志。
