---
title: Hyper-Threading Technology
top: false
cover: false
toc: true
mathjax: true
date: 2020-02-21 21:35:04
password:
summary:
tags: 
- 超线程技术
categories:
- 处理器技术
---

# Overview
---
超线程技术是Intel引入的一种同步多线程技术，在体系结构上，具有超线程技术的处理器每个核由两个逻辑处理器组成，每个逻辑处理器都有自己的体系结构状态。每个逻辑处理器可以单独停止、中断或执行指定的线程，独立于共享同一物理核心的其他逻辑处理器（officially called Hyper-Threading Technology or HT Technology and abbreviated as HTT or HT）。

Intel的超线程技术，目的是为了充分地利用一个单核CPU地资源。CPU在执行一条机器指令时，并不会完全地利用所有CPU的资源，而且实际上，是有大量资源被闲置的。超线程技术允许两个线程同时不冲突地使用CPU中的资源。比如一条整数运算指令会只用到整数运算单元，此时浮点数运算单元就空闲了，若使用超线程技术，且另一个线程刚好此时要执行一个浮点运算指令，CPU就允许属于两个不同线程的整数运算指令和浮点数运算指令同时执行，达到真正的并行。但这两个线程当前也并不意味着两个线程在同一个CPU中一直可以并行执行，只是恰好碰到两个线程当前要执行的指令不使用相同的CPU资源才可以真正的并行执行。

# CPU个数，核心数，线程数
---
* Windows通过命令查看核心数和线程数：
在cmd命令中输入“wmic”，回车；然后再输入“cpu get ”。 
NumberOfCores：表示CPU核心数 
NumberOfLogicalProcessors：表示CPU线程数 

# 参考文献
---
1. en.wikipedia.org/wiki/Hyper-threading
2. https://www.zhihu.com/question/20277695
3. https://blog.csdn.net/hong10086/article/details/81633669