# 概览

magbox是一个求解格点Landau-Lifshitz-Gilbert方程的python库，支持CPU或者GPU加速。

## 安装

magbox依赖的库为：numpy, torch, tqdm
python版本要求为3.9及以上
如果需要启用pytorch的GPU支持，需要先手动安装pytorch，详情见pytorch的[官网](https://pytorch.org/get-started/locally/ "pytorch安装")

直接通过pip安装magbox
`pip install magbox`

## 基本用法

先导入magbox

```python
import magbox
```

通过Lattice类创建一个晶格类型，使用Vars类定义相互作用大小。

```python
lt = magbox.Lattice(type="square", size=[N1, N2], periodic=True)
vars = magbox.Vars(K1=1,J=2)
```

`spin3`用于初始化磁矩，初始值`x`,`y`,`z`的总大小必须和`Lattice`中`size`所定义的总格点数一致。可以指定设备和数据类型。
`llg3`用于初始化求解器，可以设置Gilbert damping的数值 `alpha`, 旋磁比`gamma`, 模拟的总时间`T`, 输出的时间步长`dt`，温度`Temp`等参数。

```python
sp = magbox.spin3(x,y,z,lt, device='cpu',type='f64')
sf = magbox.llg3(sp,vars,alpha=0.1,gamma=1,T=50,dt=1)
```
调用求解器的`run`方法进行模拟，

```python
t, S, stats,err_info = sf.run(sp)
```

`t`是输出的时间序列。`S`是求解得到的磁矩值，第一个维度表示空间，每3个值为一组，分别代表同一个位置的xyz分量的值。第二个维度表示时间，大小和`t`的长度相同。

## 目前支持的相互作用

1. 外磁场。
2. 单轴各项异性。
3. 海森堡交换耦合。

## 开发计划

1. 支持更多的相互作用。
2. 支持更多的晶格类型。
3. 添加频域求解，直接求解能带。
4. 添加批处理支持。