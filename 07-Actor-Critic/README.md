# Actor-Critic算法框架
关于AC框架和A2C的解释，建议阅读这篇博客：https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

演员-评论家算法(Actor-Critic Algorithm)是一种结合策略梯度和时序差分学习的强化学习方法，其中：

- 演员(Actor)是指策略函数  πθ(a∣s)，即学习一个策略来得到尽量高的回报。
- 评论家(Critic)是指值函数 Vπ(s)，对当前策略的值函数进行估计，即评估演员的好坏。
- 借助于值函数，演员-评论家算法可以进行单步更新参数，不需要等到回合结束才进行更新。


## A2C
Advantage Actor-Critic


## 代码运行
- 运行main.py,选择算法（AC，A2C），开始训练
    ```shell
    python main_ddpg.py --algo [algo_name] --train 
    ```
    例如：
    ```shell
    python main_ddpg.py --algo A2C --train 
    ```
- 如果想要接着上次的训练，则再加上`--restore`
    ```shell
    python main_ddpg.py --algo [algo_name] --train --restore 
    ```
- 测试
    ```shell
    python main_ddpg.py --algo [algo_name] 
    ```
    例如：
    ```shell
    python main_ddpg.py --algo A2C 
    ```

TODO:
> 目前AC和A2C效果都很差，等以后调试