#!/usr/bin/env python3
"""
抖音评论预过滤脚本
目标：在调用LLM之前过滤掉明显无关的评论，减少LLM调用次数

过滤策略：
1. 规则过滤：空评论、纯表情、纯@、太短的无关评论
2. 关键词过滤：汽车相关关键词匹配
"""

import pandas as pd
import re
import os
import sys

# ============== 配置 ==============
INPUT_FILE = os.path.join(os.path.dirname(__file__), "抖音评论_合并.xlsx")
OUTPUT_DIR = os.path.dirname(__file__)

# 保留待分析的数据量（用于后续LLM分析）
KEEP_COUNT = 5000  # 预过滤后保留的最大数量

# ============== 关键词库 ==============
# 汽车相关关键词（评论中包含这些词才需要LLM分析）
CAR_KEYWORDS = [
    # 品牌/车型
    '车', '汽车', '车型', '智驾', '自动驾驶', '自动驾驶', '辅助驾驶', '智能驾驶',
    '续航', '电池', '充电', '充电桩', '家充', '快充', '试驾', '试乘', 
    '提车', '购车', '买车', '卖车', 'car', 'auto', 
    '华为', '理想', '问界', '蔚来', '小鹏', '比亚迪', '小米', '特斯拉', '极氪', '领克',
    'M9', 'M8', 'M7', 'M5', 'M5', 'SU7', 'U8', 'U9', 'ET7', 'ES6', 'ES8',
    '问界M9', '问界M7', '问界M5', '理想L9', '理想L8', '理想L7', '理想L6',
    '享界', '尊界', '智界', 'R7', 'S7', 'L6', 'L7', 'L9', '001', '007',
    
    # 汽车零部件/功能
    '空间', '座椅', '内饰', '外观', '油耗', '电耗', '能耗', '动力', '加速', '刹车', 
    '泊车', '自动泊车', '倒车', '泊入', '泊出',
    '变道', '并线', '超车', '高速', '城区', '乡道', '山路',
    '智能座舱', '车机', '系统', 'OTA', '升级', '更新', '版本',
    'ACC', 'LCC', 'AEB', 'NOA', 'NCA', 'NOP', 'ICC', 'LCA',
    '天窗', '后备箱', '后备', '前备箱', '电机', '四驱', '两驱', '后驱', '前驱',
    '底盘', '悬挂', '减震', '避震', '转向', '方向盘',
    '音响', '喇叭', '氛围灯', '灯语', '空调', '暖风', '制冷',
    '仪表盘', '仪表', '中控', '屏幕', 'HUD', 'AR-HUD', '抬头显示',
    '雷达', '摄像头', '传感器', '激光雷达', '毫米波',
    '辅助', '驾驶', '司机', '车主', '车友', '车群',
    
    # 评价相关
    '性价比', '价格', '售价', '落地', '贷款', '分期', '优惠', '补贴', '活动',
    '配置', '选配', '标配', '顶配', '低配', '中配',
    '质量', '品质', '做工', '细节', '材质', '用料',
    '安全', '安全性', '碰撞', '事故', '维权', '投诉', '售后', '服务',
    '保养', '维修', '保修', '质保', '维护', '保险',
    '二手', '贬值', '残值', '保值', '置换', '收购',
    '油耗子', '满意', '后悔', '推荐', '建议', '吐槽', '差评', '好评',
    '真香', '翻车', '翻牌', '打脸', '充值', '软文', '广告',
]

# 无关评论关键词（直接过滤）- 只有不含汽车关键词时才过滤
IRRELEVANT_KEYWORDS = [
    # 纯粹的情绪表达，无任何信息量（但如果含车关键词则保留）
    '前排', '占位', '沙发', 'mark', '马克', '路过', '看看',
    '无语', '服了', '醉了', '裂开', '破防', '绷不住',
    
    # 与车无关的话题（绝对无关）
    '股票', '基金', '理财', '买房', '房价', '装修', '工作', '工资', '收入',
    '找对象', '相亲', '结婚', '离婚', '孩子', '上学', '教育', '考试',
    '天气', '下雨', '下雪', '地震', '台风', '疫情', '阳了', '发烧',
    '世界杯', '足球', '篮球', 'NBA', '欧冠', '比赛', '球队', '梅西', 'C罗',
    '明星', '网红', '直播', '短视频', '抖音', '小红书', '微博',
    '电影', '电视剧', '综艺', '动漫', '游戏', '王者', '原神', '吃鸡',
    '考研', '考公', '求职', '面试', '辞职', '跳槽', '裁员',
    
    # 明显广告/引流
    '加微信', 'vx', '微信', 'qq群', '拼多多', '淘宝', '京东', '闲鱼',
    '代购', '微商', '分销', '代理', '招商', '加盟',
]


def is_pure_emoji(text):
    """检测纯表情/符号评论（评论内容只包含emoji表情，无实质文字）"""
    if not text:
        return False
    text = str(text).strip()
    if len(text) == 0:
        return True
    
    # Unicode emoji范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    
    # 移除所有emoji
    cleaned = emoji_pattern.sub('', text)
    
    # 移除标点符号和空格
    cleaned = re.sub(r'[，。！？、：；—…·～,\.!?:\'\"\-_\s\[\]（）()]+', '', cleaned)
    
    # 如果完全空了，说明只有emoji+标点+[]，算纯表情
    if len(cleaned) == 0:
        return True
    
    # 检查是否含有任何汉字或英文字母
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', cleaned))
    has_english = bool(re.search(r'[a-zA-Z]', cleaned))
    has_number = bool(re.search(r'[0-9]', cleaned))
    
    # 如果有汉字/英文/数字，就不是纯表情
    if has_chinese or has_english or has_number:
        return False
    
    return True


def is_pure_mention(text):
    """检测纯@提及评论"""
    if not text:
        return False
    text = str(text).strip()
    # 如果只有@xxx这种格式
    parts = text.split('@')
    # 移除@和其后面的内容
    remaining = ''.join(parts[1:])  # 跳过第一个空字符串
    remaining = re.sub(r'[\s\[\]（）()，。！？、：；""''《》【】]+', '', remaining)
    return len(remaining) <= 2  # @xxx后面只有很少字符


def contains_car_keyword(text):
    """检测是否包含汽车相关关键词"""
    if not text:
        return False
    text = str(text).lower()
    for kw in CAR_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def is_irrelevant(text):
    """检测明显无关的评论"""
    if not text:
        return True
    text = str(text).strip()
    if len(text) < 5:
        # 太短的评论需要进一步判断
        return not contains_car_keyword(text)
    
    # 检查无关关键词
    for kw in IRRELEVANT_KEYWORDS:
        if kw in text:
            return True
    
    return False


def filter_comments(df):
    """预过滤评论"""
    print("\n" + "="*60)
    print("[预过滤] 开始过滤抖音评论")
    print("="*60)
    
    total = len(df)
    print(f"原始评论数: {total:,}")
    
    # 步骤1: 过滤空评论
    df = df[df['评论内容'].notna()].copy()
    df = df[df['评论内容'].str.strip() != '']
    empty_filtered = total - len(df)
    print(f"步骤1 - 空评论过滤: -{empty_filtered:,} 条")
    
    if len(df) == 0:
        return df
    
    # 步骤2: 过滤纯表情评论
    pure_emoji_mask = df['评论内容'].apply(is_pure_emoji)
    pure_emoji_count = pure_emoji_mask.sum()
    df = df[~pure_emoji_mask].copy()
    print(f"步骤2 - 纯表情过滤: -{pure_emoji_count:,} 条")
    
    if len(df) == 0:
        return df
    
    # 步骤3: 过滤纯@提及
    pure_mention_mask = df['评论内容'].apply(is_pure_mention)
    pure_mention_count = pure_mention_mask.sum()
    df = df[~pure_mention_mask].copy()
    print(f"步骤3 - 纯@提及过滤: -{pure_mention_count:,} 条")
    
    if len(df) == 0:
        return df
    
    # 步骤4: 关键词过滤 - 只保留汽车相关的
    car_related_mask = df['评论内容'].apply(contains_car_keyword)
    car_related_count = car_related_mask.sum()
    non_car_related = len(df) - car_related_count
    
    print(f"步骤4 - 关键词过滤:")
    print(f"       汽车相关: {car_related_count:,} 条 ({car_related_count/len(df)*100:.1f}%)")
    print(f"       非汽车相关: {non_car_related:,} 条 ({non_car_related/len(df)*100:.1f}%)")
    
    # 保留汽车相关的评论
    df_filtered = df[car_related_mask].copy().reset_index(drop=True)
    
    # 如果过滤后数量太多，随机采样
    if len(df_filtered) > KEEP_COUNT:
        print(f"\n步骤5 - 随机采样: {len(df_filtered):,} -> {KEEP_COUNT:,} 条")
        df_filtered = df_filtered.sample(n=KEEP_COUNT, random_state=42).reset_index(drop=True)
    
    print(f"\n[结果] 预过滤后待分析: {len(df_filtered):,} 条 "
          f"(原始的 {len(df_filtered)/total*100:.1f}%)")
    print(f"       预计减少LLM调用: {total - len(df_filtered):,} 次")
    
    return df_filtered


def main():
    """主函数"""
    print("="*60)
    print("[抖音评论预过滤工具]")
    print("="*60)
    
    # 读取数据
    print(f"\n读取数据: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"[错误] 文件不存在: {INPUT_FILE}")
        sys.exit(1)
    
    df = pd.read_excel(INPUT_FILE)
    print(f"读取完成: {len(df):,} 条评论")
    
    # 预过滤
    df_filtered = filter_comments(df)
    
    if len(df_filtered) == 0:
        print("\n[警告] 过滤后无评论")
        return
    
    # 保存过滤后的数据
    output_file = os.path.join(OUTPUT_DIR, "抖音评论_预过滤后.xlsx")
    df_filtered.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n已保存: {output_file}")
    
    # 保存统计报告
    report_file = os.path.join(OUTPUT_DIR, "预过滤报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("抖音评论预过滤报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"原始评论数: {len(df):,}\n")
        f.write(f"过滤后评论数: {len(df_filtered):,}\n")
        f.write(f"过滤比例: {(1-len(df_filtered)/len(df))*100:.1f}%\n")
        f.write(f"预计节省LLM调用: {len(df) - len(df_filtered):,} 次\n")
    
    print(f"已保存报告: {report_file}")
    print("\n[完成] 预过滤完成!")


if __name__ == "__main__":
    main()
