**`关于链家全网房价数据分析挖掘项目`**

**关于该项目**  
本项目是在大数据分析文摘的学习基础之上，结合个人掌握技能和知识进行的时间挖掘实践。
项目部分参考了其他工程师的指导，在此作简单分享，请多指正。

**数据说明**
1. 数据信息：
 - 数据量：40多万
 - 时间：2018年7月前
2. 来源
 - 作者：田昕峣
 - 获取方式：https://github.com/XinyaoTian/lianjia_Spider  

**项目目标**  
建立单位面积房价的预测模型

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#数据导入" data-toc-modified-id="数据导入-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据导入</a></span></li><li><span><a href="#数据探索：" data-toc-modified-id="数据探索：-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>数据探索：</a></span></li><li><span><a href="#数据处理：" data-toc-modified-id="数据处理：-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>数据处理：</a></span></li><li><span><a href="#绘图分析" data-toc-modified-id="绘图分析-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>绘图分析</a></span><ul class="toc-item"><li><span><a href="#单变量观察" data-toc-modified-id="单变量观察-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>单变量观察</a></span><ul class="toc-item"><li><span><a href="#面积、查看次数、收藏次数、发布时间" data-toc-modified-id="面积、查看次数、收藏次数、发布时间-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>面积、查看次数、收藏次数、发布时间</a></span></li><li><span><a href="#2年产权、5年产权、房屋户型" data-toc-modified-id="2年产权、5年产权、房屋户型-4.1.2"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>2年产权、5年产权、房屋户型</a></span></li><li><span><a href="#朝向、装修程度、电梯配备、楼层位置、楼型、建成时间" data-toc-modified-id="朝向、装修程度、电梯配备、楼层位置、楼型、建成时间-4.1.3"><span class="toc-item-num">4.1.3&nbsp;&nbsp;</span>朝向、装修程度、电梯配备、楼层位置、楼型、建成时间</a></span></li></ul></li><li><span><a href="#多维度分析" data-toc-modified-id="多维度分析-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>多维度分析</a></span><ul class="toc-item"><li><span><a href="#产权和查看次数、收藏次数" data-toc-modified-id="产权和查看次数、收藏次数-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>产权和查看次数、收藏次数</a></span></li><li><span><a href="#户型+产权和查看次数、收藏次数¶" data-toc-modified-id="户型+产权和查看次数、收藏次数¶-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>户型+产权和查看次数、收藏次数¶</a></span></li></ul></li><li><span><a href="#关联分析" data-toc-modified-id="关联分析-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>关联分析</a></span><ul class="toc-item"><li><span><a href="#2年产权、5年产权vs装修程度、户型" data-toc-modified-id="2年产权、5年产权vs装修程度、户型-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>2年产权、5年产权vs装修程度、户型</a></span></li><li><span><a href="#电梯、楼层、楼型、建成时间单变量统计" data-toc-modified-id="电梯、楼层、楼型、建成时间单变量统计-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>电梯、楼层、楼型、建成时间单变量统计</a></span></li><li><span><a href="#连续变量的相关性" data-toc-modified-id="连续变量的相关性-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>连续变量的相关性</a></span></li><li><span><a href="#异常值检查（size_house_edit1与smeter_price_edit1关系为例）" data-toc-modified-id="异常值检查（size_house_edit1与smeter_price_edit1关系为例）-4.3.4"><span class="toc-item-num">4.3.4&nbsp;&nbsp;</span>异常值检查（size_house_edit1与smeter_price_edit1关系为例）</a></span><ul class="toc-item"><li><span><a href="#检查" data-toc-modified-id="检查-4.3.4.1"><span class="toc-item-num">4.3.4.1&nbsp;&nbsp;</span>检查</a></span></li><li><span><a href="#剔除异常点" data-toc-modified-id="剔除异常点-4.3.4.2"><span class="toc-item-num">4.3.4.2&nbsp;&nbsp;</span>剔除异常点</a></span></li></ul></li></ul></li><li><span><a href="#目标变量处理——满足整体分布" data-toc-modified-id="目标变量处理——满足整体分布-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>目标变量处理——满足整体分布</a></span><ul class="toc-item"><li><span><a href="#绘制正态分布图" data-toc-modified-id="绘制正态分布图-4.4.1"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>绘制正态分布图</a></span></li><li><span><a href="#绘制QQ图" data-toc-modified-id="绘制QQ图-4.4.2"><span class="toc-item-num">4.4.2&nbsp;&nbsp;</span>绘制QQ图</a></span></li><li><span><a href="#变换处理与查看" data-toc-modified-id="变换处理与查看-4.4.3"><span class="toc-item-num">4.4.3&nbsp;&nbsp;</span>变换处理与查看</a></span></li></ul></li><li><span><a href="#缺失值处理" data-toc-modified-id="缺失值处理-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>缺失值处理</a></span></li><li><span><a href="#其它特征工程" data-toc-modified-id="其它特征工程-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>其它特征工程</a></span><ul class="toc-item"><li><span><a href="#1、有许多特征实际上是类别型的特征，但给出来的是数字，所以需要将其转换成类别型。" data-toc-modified-id="1、有许多特征实际上是类别型的特征，但给出来的是数字，所以需要将其转换成类别型。-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>1、有许多特征实际上是类别型的特征，但给出来的是数字，所以需要将其转换成类别型。</a></span></li><li><span><a href="#2、接下来-LabelEncoder，对部分类别的特征进行编号。" data-toc-modified-id="2、接下来-LabelEncoder，对部分类别的特征进行编号。-4.6.2"><span class="toc-item-num">4.6.2&nbsp;&nbsp;</span>2、接下来 LabelEncoder，对部分类别的特征进行编号。</a></span></li><li><span><a href="#3、检查变量的正态分布情况" data-toc-modified-id="3、检查变量的正态分布情况-4.6.3"><span class="toc-item-num">4.6.3&nbsp;&nbsp;</span>3、检查变量的正态分布情况</a></span><ul class="toc-item"><li><span><a href="#检查" data-toc-modified-id="检查-4.6.3.1"><span class="toc-item-num">4.6.3.1&nbsp;&nbsp;</span>检查</a></span></li><li><span><a href="#变换处理" data-toc-modified-id="变换处理-4.6.3.2"><span class="toc-item-num">4.6.3.2&nbsp;&nbsp;</span>变换处理</a></span></li></ul></li><li><span><a href="#哑变量处理" data-toc-modified-id="哑变量处理-4.6.4"><span class="toc-item-num">4.6.4&nbsp;&nbsp;</span>哑变量处理</a></span></li></ul></li></ul></li><li><span><a href="#建立模型" data-toc-modified-id="建立模型-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>建立模型</a></span><ul class="toc-item"><li><span><a href="#数据准备" data-toc-modified-id="数据准备-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>数据准备</a></span></li><li><span><a href="#模型函数" data-toc-modified-id="模型函数-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>模型函数</a></span><ul class="toc-item"><li><span><a href="#模型函数设定" data-toc-modified-id="模型函数设定-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>模型函数设定</a></span><ul class="toc-item"><li><span><a href="#lasso模型" data-toc-modified-id="lasso模型-5.2.1.1"><span class="toc-item-num">5.2.1.1&nbsp;&nbsp;</span>lasso模型</a></span></li><li><span><a href="#ENet模型" data-toc-modified-id="ENet模型-5.2.1.2"><span class="toc-item-num">5.2.1.2&nbsp;&nbsp;</span>ENet模型</a></span></li><li><span><a href="#KRR模型" data-toc-modified-id="KRR模型-5.2.1.3"><span class="toc-item-num">5.2.1.3&nbsp;&nbsp;</span>KRR模型</a></span></li><li><span><a href="#GBoost模型" data-toc-modified-id="GBoost模型-5.2.1.4"><span class="toc-item-num">5.2.1.4&nbsp;&nbsp;</span>GBoost模型</a></span></li><li><span><a href="#xgboost模型" data-toc-modified-id="xgboost模型-5.2.1.5"><span class="toc-item-num">5.2.1.5&nbsp;&nbsp;</span>xgboost模型</a></span></li><li><span><a href="#LightGBM模型" data-toc-modified-id="LightGBM模型-5.2.1.6"><span class="toc-item-num">5.2.1.6&nbsp;&nbsp;</span>LightGBM模型</a></span></li></ul></li><li><span><a href="#模型得分" data-toc-modified-id="模型得分-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>模型得分</a></span></li></ul></li><li><span><a href="#模型融合" data-toc-modified-id="模型融合-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>模型融合</a></span><ul class="toc-item"><li><span><a href="#基模型融合" data-toc-modified-id="基模型融合-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>基模型融合</a></span></li><li><span><a href="#构建stacking-averagd-models的类" data-toc-modified-id="构建stacking-averagd-models的类-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>构建stacking averagd models的类</a></span></li><li><span><a href="#测试模型融合" data-toc-modified-id="测试模型融合-5.3.3"><span class="toc-item-num">5.3.3&nbsp;&nbsp;</span>测试模型融合</a></span><ul class="toc-item"><li><span><a href="#stacking" data-toc-modified-id="stacking-5.3.3.1"><span class="toc-item-num">5.3.3.1&nbsp;&nbsp;</span>stacking</a></span></li><li><span><a href="#xgboost" data-toc-modified-id="xgboost-5.3.3.2"><span class="toc-item-num">5.3.3.2&nbsp;&nbsp;</span>xgboost</a></span></li><li><span><a href="#lightgbm" data-toc-modified-id="lightgbm-5.3.3.3"><span class="toc-item-num">5.3.3.3&nbsp;&nbsp;</span>lightgbm</a></span></li></ul></li><li><span><a href="#结果" data-toc-modified-id="结果-5.3.4"><span class="toc-item-num">5.3.4&nbsp;&nbsp;</span>结果</a></span></li></ul></li></ul></li></ul></div>

