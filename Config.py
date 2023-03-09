inception_layers = 4       # AFCM模块数量
use_our_model = True       # 是否使用我们的模型
use_physics = False         # 是否使用物理约束
epoch_nums = 20000         # 训练轮数
draw_figure = False         # 绘制图表
open_cosine = False         # 开启余弦退火
use_our_dataSets = False    # 使用我们做的数据集
mix_all_data = True        # 混合（0,140W）和（0， 23W）数据后重新分配训练和验证集。
load_static = ""            # 加载模型做微调
use_Kpp = True              # 添加KPP物理意义


