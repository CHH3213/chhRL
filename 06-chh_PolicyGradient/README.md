# 策略梯度方法
策略梯度的思想是:我们不是去参数化值函数并做贪婪策略改进，而是参数化策略并做梯度下降到一个方向来改进它。
（因为是要最大化奖励，所以这边其实是做梯度上升，意思是一样的）

## REINFORCE算法 
- REINFORCE算法是属于Monte Carlo Policy Gradient，REINFORCE 用的是回合更新的方式。
- 它在代码上的处理是先拿到每个步骤的奖励，然后计算每个步骤的未来总收益G_t是多少，然后拿每个G_t代入公式，去优化每一个动作的输出。
- 以下为伪代码:
![在这里插入图片描述](https://img-blog.csdnimg.cn/9fc7ea117a0e479a91f20690e7efd07d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ0hIMzIxMw==,size_20,color_FFFFFF,t_70,g_se,x_16)


### REINFORCE算法离散形式
- 测试环境为CartPole-v0，代码文件为：REINFORCE_discrete.py
### REINFORCE算法连续形式
- 测试环境为Pendulum-v0，代码文件为：REINFORCE_continuous.py
- 需要在主函数main.py中，修改环境参数：
    ```python
        parser.add_argument('--env_name', default="Pendulum-v0",
                            help="environment name：CartPole-v0/Pendulum-v0")
    ```
- 目前还有问题，训练不出来
### REINFORCE_with_Baseline
- 只写了离散情况的使用baseline的REINFORCE算法
- 测试环境为CartPole-v0
- 代码文件为：REINFORCE_with_Baseline.py
- 需要在主函数main.py中，修改参数：
    ```python
        parser.add_argument('--baseline', default=True, action="store_true",
                            help="use REINFORCE with baseline if true")
    ```
