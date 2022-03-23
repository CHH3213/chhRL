
# [Proximal Policy Optimization](https://blog.csdn.net/weixin_42301220/article/details/123123261)
近端策略优化(Proximal Policy Optimization，简称 PPO) 是 policy gradient 的一个变形，它是现在 OpenAI 默认的强化学习算法。

Openai Baselines 博客: https://blog.openai.com/openai-baselines-ppo/

论文参考：https://arxiv.org/abs/1707.06347


![在这里插入图片描述](https://img-blog.csdnimg.cn/440e84e7e56b4394a1469716631d0739.png)




![图片](https://user-images.githubusercontent.com/62095277/156288810-96ef9ea9-96ae-45ea-a15a-8194751cdcfa.png)

## PPO-Penalty(PPO1)
![在这里插入图片描述](https://img-blog.csdnimg.cn/49665bf6b73c448cbf64744c6b92903c.png)



![图片](https://user-images.githubusercontent.com/62095277/156288757-f766bd29-dd83-4112-aafa-607cc319f3fb.png)

## PPO-Clip(PPO2)

![在这里插入图片描述](https://img-blog.csdnimg.cn/49a31026289c4a2faa9ec0321dae7973.png)

![图片](https://user-images.githubusercontent.com/62095277/156288779-cbd6a867-afd5-433a-8b12-aae1536f61a6.png)



## 代码实现
- 运行主函数main.py，开始训练或测试
    ```shell
    python main.py --algo [algo_name]  --env_name [env_name] --train
    ```
  algo_name可选：ppo_penalty，ppo_clip.目前ppo_penalty只实现在离散环境即CartPole-v0中。

  env_name可选：Pendulum-v0,CartPole-v0。其中，Pendulum-v0为连续动作环境，CartPole-v0为离散动作环境。

  如：
    ```shell
    python main.py --algo ppo_penalty  --env_name CartPole-v0 --train
    ```
- 如果想要接着上次的训练，则再加上`--restore`
    ```shell
    python main.py --algo ppo_penalty  --env_name CartPole-v0 --train --restore
    ```
- 训练完后测试：将`--train`去除即可。
  
  如：
    ```shell
    python main.py --algo ppo_penalty  --env_name CartPole-v0 
    ```
  

  目前在连续环境下，PPO的效果还没出来，需要后期再调试。