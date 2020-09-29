# /********************************************************************************
# **  Author:     吴崇益
# **  Establish:  2020-09-29
# **  Summary:    断约客户回归预测
# *********************************************************************************/

"""
说明：
    冬眠客户：套餐最晚到期时间超过180天的客户
    断约客户：30天<=套餐最晚到期时间<=180 的客户
    回归定义：服务开始时间 在 未来30天（一个月）内
    客户限制：直销43城市住宅经纪人
    举例说明：时间【20200731】；断约时间段【180天前-30天前】；回归时间段【20200801至20200831】
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier


class BrokeContractCustomerBackML(object):
    def __init__(self):
        self._states = {"cal_dt": "日期",
                        "broker_id": "用户ID",
                        "city_name_fang": "城市",
                        "order_loss_days": "断约天数",
                        "order_category": "断约套餐类型",
                        "order_is_self_pay": "是否个人购买",
                        "order_is_zf_pure": "是否纯租套餐",
                        "base_agent_alone": "是否独立经纪人",
                        "base_broker_status": "账号是否注销",
                        "base_service_year": "服务年限",
                        "consume_money_30": "近30天消费金额",
                        "order_num_365": "近一年订单数量",
                        "house_modify_days_30": "近30天内编辑房源天数",
                        "house_last_modify_days": "最近一次编辑房源间隔",
                        "re_jf_days_90": "近90天被禁发天数",
                        "re_jf_times_90": "近90天被禁发次数",
                        "app_login_days_30": "近30天app登陆天数",
                        "app_enter_times_30": "近30天app登陆次数",
                        "app_stay_mins_30": "近30天app登陆停留时长",
                        "app_last_login_days": "最近一次app登陆间隔",
                        "pc_login_days_30": "近30天pc登陆天数",
                        "pc_enter_times_30": "近30天pc登陆次数",
                        "pc_last_login_days": "最近一次pc登陆间隔",
                        "re_mass_agent_number_all": "所属公司体量",
                        "re_mass_cooperation_rate": "所属公司合作率",
                        "wl_u2b_sessions_30": "近30天微聊u2b客户数",
                        "wl_u2b_reply_rate_30": "近30天微聊u2b回复率",
                        "wl_u2b_reply_5min_rate_30": "近30天微聊u2b5min回复率",
                        "wl_b2u_sessions_30": "近30天微聊b2u客户数",
                        "wl_b2u_reply_rate_30": "近30天微聊b2u回复率",
                        "tel_u2b_call_times_30": "近30天来电客户数",
                        "tel_u2b_call_success_rate_30": "近30天来电成功接听率",
                        "tel_avg_call_times_30": "近30天来电平均每次通话时长",
                        "wl_avg_company_num_30": "所属公司近30天会员微聊获客数",
                        "wl_avg_store_num_30": "所属门店近30天会员微聊获客数",
                        "tel_avg_company_num_30": "所属公司近30天会员电话获客数",
                        "tel_avg_store_num_30": "所属门店近30天会员电话获客数",
                        "visit_num_tel_day_30": "近30天电话拜访天数",
                        "visit_avg_num_tel_30": "最近30天内日均门店拜访次数",
                        "visit_num_store_day_30": "近30天门店拜访天数",
                        "visit_avg_num_store_30": "最近30天内日均电话拜访次数",
                        "target": "目标值"
                        }
        self.city_rise = None
        self.cate_rise = None
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = pd.read_table('./source.txt', encoding='utf-8', sep='\t')
        # 指标中文化
        self.df.rename(columns=self._states, inplace=True)

    def feature_deal(self):
        # 删除强关联指标&异常指标
        del_feature = [
            '近90天被禁发次数',
            '最近一次app登陆间隔',
            '最近一次pc登陆间隔',
            '近30天pc登陆天数',
            '近30天微聊u2b5min回复率',
            '近30天来电平均每次通话时长',
            '所属公司近30天会员电话获客数',
            '所属门店近30天会员电话获客数',
            '近30天电话拜访天数',
            '近30天门店拜访天数'
        ]
        self.df = self.df.drop(del_feature, axis=1)

        # 近30天来电客户数 与 近30天微聊u2b客户数 强相关-->变动为 近30天wl2tel客户转换率
        self.df['近30天来电客户数'] = self.df['近30天来电客户数'] / (self.df['近30天微聊u2b客户数'])
        self.df.rename(columns={'近30天来电客户数': '近30天wl2tel客户转换率'}, inplace=True)

        # 特征离散化：城市、套餐类型
        city_tmp = self.df['城市']
        self.city_rise = pd.get_dummies(city_tmp, prefix="城市")
        cate_tmp = self.df['断约套餐类型']
        self.cate_rise = pd.get_dummies(cate_tmp, prefix="断约套餐类型")
        pass

    def model_train(self):
        # 特征值、目标值
        train_feature = ["断约天数",
                         "是否个人购买",
                         "是否纯租套餐",
                         "是否独立经纪人",
                         "账号是否注销",
                         "服务年限",
                         "近30天消费金额",
                         "近一年订单数量",
                         "近30天内编辑房源天数",
                         "最近一次编辑房源间隔",
                         "近90天被禁发天数",
                         "近30天app登陆天数",
                         "近30天app登陆次数",
                         "近30天app登陆停留时长",
                         "近30天pc登陆次数",
                         "所属公司体量",
                         "所属公司合作率",
                         "近30天微聊u2b客户数",
                         "近30天微聊u2b回复率",
                         "近30天微聊b2u客户数",
                         "近30天微聊b2u回复率",
                         "近30天wl2tel客户转换率",
                         "近30天来电成功接听率",
                         "所属公司近30天会员微聊获客数",
                         "所属门店近30天会员微聊获客数",
                         "最近30天内日均门店拜访次数",
                         "最近30天内日均电话拜访次数"]
        X, y = pd.concat([self.df[train_feature], self.cate_rise, self.city_rise], axis=1), self.df['目标值']
        # 数据集切分, random_state=666
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

        self.clf = XGBClassifier(
            learning_rate=0.01,  # 在更新中使用步长收缩以防止过度拟合。在每个增强步骤之后，我们都可以直接获得新特征的权重，并eta缩小特征权重以使增强过程更加保守。
            n_estimators=3000,  # 树的个数
            gamma=0.2,  # 在树的叶子节点上进行进一步分区所需的最小损失减少。越大gamma，算法将越保守。
            max_depth=10,  # 一棵树的最大深度。增加此值将使模型更复杂，并且更可能过度拟合
            min_child_weight=3,  # 子级中所需实例重量的最小总和（hessian）
            subsample=0.6,  # (0,1]训练实例的子样本比率。将其设置为0.5意味着XGBoost将在树木生长之前随机采样一半的训练数据。这样可以防止过度拟合。二次采样将在每个增强迭代中进行一次
            colsample_bytree=1,  # 构造每棵树时列的子样本比率。对每一个构造的树进行一次二次采样。
            reg_lambda=2.4,  # L2正则化权重项。增加此值将使模型更加保守。
            reg_alpha=2.6  # L1正则化权重项。增加此值将使模型更加保守。
        )
        self.clf.fit(self.X_train, self.y_train, eval_metric='aucpr')
        pass

    def print_feature_imp(self):
        importance = self.clf.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = self.X_train.columns
        with open('./result.txt', encoding='utf-8', mode='w') as file:
            for f in range(self.X_train.shape[1]):
                file.write("(%2d)\t%-30s\t%.4f\n" % (f + 1, features[indices[f]], importance[indices[f]] * 10000))
                print("(%2d)\t%-30s\t%.4f" % (f + 1, features[indices[f]], importance[indices[f]] * 10000))
        # 作图
        plt.figure(figsize=(15, 8))
        plt.title('Feature Importance')
        plt.bar(range(self.X_train.shape[1]), importance[indices], color='blue')
        plt.xticks(range(self.X_train.shape[1]), indices)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.show()

    def print_null(self):
        # 各字段缺失比例计算
        na_rate = (len(self.df) - self.df.count()) / len(self.df)
        na_rate_order = na_rate.sort_values(ascending=False)
        with open('./缺失值排行.txt', encoding='utf-8', mode='w') as file:
            file.write(na_rate_order)

    def print_model_score(self):
        # 模型评估分数
        y_true, y_pred = self.y_test, self.clf.predict(self.X_test)
        print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
        print("roc_auc_score : %.4g" % metrics.roc_auc_score(y_true, y_pred))
        print("precision_score : %.4g" % metrics.precision_score(y_true, y_pred))
        print("recall_score : %.4g" % metrics.recall_score(y_true, y_pred))
        print("f1_score : %.4g" % metrics.f1_score(y_true, y_pred))
        pass


if __name__ == '__main__':
    # 实例化对象
    BrokeContractCustomerBackML = BrokeContractCustomerBackML()
    # 数据预处理
    BrokeContractCustomerBackML.feature_deal()
    # 数据切分、模型训练
    BrokeContractCustomerBackML.model_train()
    # 输出结果
    BrokeContractCustomerBackML.print_feature_imp()
    BrokeContractCustomerBackML.print_null()
    BrokeContractCustomerBackML.print_model_score()
