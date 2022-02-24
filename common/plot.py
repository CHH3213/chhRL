# --*--coding: utf-8 --*--
"""
Created with IntelliJ PyCharm.
@Time: 2022/2/23 23:42
@PROJECT_NAME: chhRL
@File: plot
@Author: chh3213
@Email:
@Description:
    画奖励图
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    """
    画训练奖励图
    :param rewards:
    :param ma_rewards:
    :param plot_cfg: 画图参数
    :param tag:
    :return:
    """
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()
