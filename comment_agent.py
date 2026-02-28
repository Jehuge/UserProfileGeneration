#!/usr/bin/env python3
"""
评论分析Agent
支持批量并行处理，每批次自动合并原始数据并追加保存
技术维度多列展开：智驾/续航/动力/底盘/能耗/空间/内饰/外观/音响/充电/安全/舒适
技术点摘要列由Python派生（零幻觉）
"""
import pandas as pd
import json
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama

# ============== 配置 ==============
DATA_DIR = ""   # 数据文件所在目录，留空代表当前目录
OUTPUT_DIR = "" # 结果输出目录，留空代表当前目录
# MODEL_NAME = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
MODEL_NAME = "hf.co/unsloth/Qwen3-4B-GGUF:Q6_K"
OLLAMA_MODEL = None

# ============================================================
# 性能配置（RTX 5070 12GB 评估）
# ============================================================
# 模型显存占用:
#   Qwen3-4B Q6_K   ≈ 3.5GB 权重
#   Qwen3-8B Q4_K_M ≈ 5.5GB 权重
#
# KV Cache（num_ctx=8192，单请求）:
#   Qwen3-4B ≈ 0.4GB  → 剩余 8.5GB / 0.4 ≈ 可跑 ~20 路（瓶颈在调度，取 8）
#   Qwen3-8B ≈ 0.8GB  → 剩余 6.5GB / 0.8 ≈ 可跑  ~8 路（建议取 4）
#
# 注意：BATCH_SIZE=5 时每批输出约 800 tokens（16 字段×5条），
# num_ctx=8192 完全够用，无需 16384（节省 KV 显存提升并发）
#
# 推荐:  4B 模型 → MAX_WORKERS=8, BATCH_SIZE=5
#        8B 模型 → MAX_WORKERS=4, BATCH_SIZE=5
# ============================================================
MAX_WORKERS = 8
# 7维度精简后每批输出量大幅减少：5条×(4基础+7维度) ≈ 320 output tokens
# 500字×5条输入 ≈ 1750 tokens + prompt 500 + output 320 = ~2570，8192完全够用
BATCH_SIZE  = 5      # 5条/批（长评论500字+7维度，8192内安全）
NUM_CTX     = 8192   # 7维度精简后恢复8192，节省每请求~0.5GB KV显存，并发更高

MAX_COMMENTS = None  # 限制处理条数，None=全部

# ============== 技术维度定义 ==============
# 每个维度在输出中独立成列，值为：正面 / 负面 / 未涉及
# 合并逻辑：续航+能耗+充电→续航补能；底盘+舒适+音响→驾乘体验；外观+内饰→外观内饰
# 维度合并精简为6个
TECH_DIMS = ['智驾', '续航补能', '动力', '驾乘体验', '空间', '外观内饰']
TECH_DIM_DESCS = [dim + '点' for dim in TECH_DIMS]

# ============== 提示词模板 ==============
# 维度字段默认"未涉及"，技术点描述字段默认""
_TECH_JSON_FIELDS = (
    ",".join(f'"{d}":"无"' for d in TECH_DIMS) + "," +
    ",".join(f'"{d}":""' for d in TECH_DIM_DESCS)
)

BATCH_ANALYSIS_PROMPT = """批量分析以下汽车评论，每条输出指定字段。

基础字段：
- 有效无效：有效/无效（纯表情/乱码/无意义重复→无效）
- 正负面：正面/负面/中性
- 营销手段：是/否（广告/软文/引流/水军→是）
- 技术亮点：是/否（主动描述技术特性→是；仅提问/闲聊→否）

技术维度（值只能是：正面/负面/无）：
- 智驾：辅助驾驶/NOA/领航/自动泊车/变道辅助
- 续航补能：续航里程/CLTC/电耗/充电速度/快充/补能
- 动力：加速/提速/扭矩/马力/动力响应
- 驾乘体验：底盘/悬挂/NVH/隔音/音响/乘坐舒适
- 空间：前后排/头腿部空间/后备箱
- 外观内饰：外观设计/内饰材质/做工/中控屏/HUD/座椅

技术点描述（对应6维度，简短描述评论对该维度的具体观点或问题，未提及填""）：
字段：智驾点/续航补能点/动力点/驾乘体验点/空间点/外观内饰点
- 要写具体观点，而非仅列关键词：如"智驾多次出错"而非"智能辅助驾驶"；"实测续航只有标称70%"而非"续航"
- 内容来自评论原文，不得添加未出现的信息

判定规则：
1. 原文有明确正/负态度→正面或负面；仅提问→无；未提及→无
2. 禁止凭车型/品牌常识推断；禁止因整体正面把所有维度填正面
3. 不确定→无（宁缺勿多）
4. 若某维度技术点描述非空，该维度值必须为正面或负面，不得为无

评论列表（序号|内容）：
{comments}

仅返回JSON数组，不含其他内容：
[{{"序号":N,"有效无效":"","正负面":"","营销手段":"","技术亮点":"",""" + _TECH_JSON_FIELDS + """}}]
/no_think"""


def load_data(limit=None, source=None):
    """加载指定来源的评论数据"""
    print("="*60)
    print("正在加载数据...")
    print("="*60)
    all_data = []
    sources_info = []

    if source is None or source == 'dongche':
        file_path = os.path.join(DATA_DIR, "懂车帝评论.xlsx")
        if os.path.exists(file_path):
            xlsx = pd.ExcelFile(file_path)
            print(f"\n[懂车帝] 共{len(xlsx.sheet_names)}个sheet:")
            for sheet in xlsx.sheet_names:
                df = xlsx.parse(sheet)
                df['车型'] = sheet.strip()
                df['来源'] = '懂车帝'
                all_data.append(df)
                print(f"   {sheet}: {len(df)} 条")
            sources_info.append(('懂车帝', len(xlsx.sheet_names)))
        else:
            print(f"\n[警告] 未找到文件: {file_path}")

    if source is None or source == 'douyin':
        file_path = os.path.join(DATA_DIR, "抖音评论.xlsx")
        if os.path.exists(file_path):
            xlsx = pd.ExcelFile(file_path)
            print(f"\n[抖音] 共{len(xlsx.sheet_names)}个sheet:")
            for sheet in xlsx.sheet_names:
                df = xlsx.parse(sheet)
                df['车型'] = sheet.strip()
                df['来源'] = '抖音'
                cols = ['昵称', '评论内容']
                if '地区' in df.columns: cols.append('地区')
                if '点赞数' in df.columns: cols.append('点赞数')
                if '评论时间' in df.columns: cols.append('评论时间')
                df = df[cols + ['车型', '来源']]
                all_data.append(df)
                print(f"   {sheet}: {len(df)} 条")
            sources_info.append(('抖音', len(xlsx.sheet_names)))
        else:
            print(f"\n[警告] 未找到文件: {file_path}")

    if not all_data:
        print("未加载到任何数据，请检查文件路径。")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['评论内容'] = df['评论内容'].fillna('').astype(str).str.strip()
    df = df[df['评论内容'] != ''].reset_index(drop=True)

    total_original = len(df)
    if limit and limit < total_original:
        df = df.head(limit)

    print(f"\n{'='*60}")
    print(f"[数据汇总]")
    for name, cnt in sources_info:
        print(f"   {name}车型: {cnt} 个")
    print(f"   原始总数: {total_original} 条")
    print(f"   实际处理: {len(df)} 条")
    if limit:
        print(f"   (已限制为前 {limit} 条)")
    print("="*60)
    return df


def sanitize_comment(comment: str, max_len: int = 500) -> str:
    """净化评论文本，避免特殊字符破坏模型输出的 JSON 结构
    max_len=500 覆盖长篇评论（200字被截断会丢失技术信息）
    如果评论数据普遍极短，可调低以提升批次吞吐量
    """
    text = comment[:max_len]
    text = text.replace('"', '\u201c').replace('\u201d', '\u201c').replace('"', '\u201c')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\\', '/')
    return text.strip()


def repair_json(text: str):
    """尝试修复常见 JSON 格式错误后重新解析"""
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return None
    raw = match.group()
    raw = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group().replace('\n', ' ').replace('\r', ' '), raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    results = []
    for obj_str in re.finditer(r'\{[^{}]*\}', raw):
        try:
            results.append(json.loads(obj_str.group()))
        except Exception:
            pass
    return results if results else None


def build_tech_summary(rec: dict) -> str:
    """
    拼接各维度技术点描述字段，生成技术点摘要。
    内容直接来自评论原文的技术描述，反映实际提及的技术点。
    示例: "NOA表现不错；CLTC400实测只有280km；底盘隔振一般"
    """
    parts = []
    for dim in TECH_DIMS:
        desc = rec.get(dim + '点', '').strip()
        if desc:
            parts.append(desc)
    return '；'.join(parts)


def post_process_results(results: list) -> list:
    """
    结果后处理（Python端校验，不依赖模型）:
    1. 标准化维度值（只保留合法值，其他归一为未涉及）
    2. 拼接技术点摘要（来自模型输出的描述字段）
    3. 自动修正矛盾: 维度有值 但 技术亮点=否 → 改为是
    4. 自动修正矛盾: 技术亮点=是 但 所有维度=未涉及 → 改为否
    """
    valid_dim_values = {'正面', '负面', '无'}
    for rec in results:
        # 标准化维度值，防止模型输出非法值
        for dim in TECH_DIMS:
            if rec.get(dim) not in valid_dim_values:
                rec[dim] = '无'

        # 一致性修正：desc非空但dim=无时，将dim设为整体正负面（兜底）
        overall = rec.get('正负面', '中性')
        fallback = '正面' if overall == '正面' else ('负面' if overall == '负面' else '正面')
        for dim in TECH_DIMS:
            if rec.get(dim + '点', '').strip() and rec.get(dim) == '无':
                rec[dim] = fallback

        rec['技术点摘要'] = build_tech_summary(rec)

        # has_tech：维度有情感值 或 有非空描述（两者取或，避免模型漏填维度值）
        has_tech = (
            any(rec.get(dim) in ('正面', '负面') for dim in TECH_DIMS) or
            any(rec.get(dim + '点', '').strip() for dim in TECH_DIMS)
        )
        if has_tech and rec.get('技术亮点') == '否':
            rec['技术亮点'] = '是'
        elif not has_tech and rec.get('技术亮点') == '是':
            rec['技术亮点'] = '否'
    return results


def make_fail_record(idx):
    """生成失败占位记录（含所有输出字段，防止merge后出现空列）"""
    rec = {
        "序号": idx, "有效无效": "未知", "正负面": "中性",
        "营销手段": "否", "技术亮点": "否", "技术点摘要": ""
    }
    for dim in TECH_DIMS:
        rec[dim] = "无"
    for dim_desc in TECH_DIM_DESCS:
        rec[dim_desc] = ""
    return rec


def _call_model_once(comments_with_idx) -> list | None:
    """调用模型一次，返回解析后的结果列表，失败返回None"""
    comments_text = "\n".join(
        f"{idx}|{sanitize_comment(comment)}" for idx, comment in comments_with_idx
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user',
                        'content': BATCH_ANALYSIS_PROMPT.format(comments=comments_text)}],
            options={'temperature': 0.1, 'num_ctx': NUM_CTX},
        )
        result_text = response['message']['content'].strip()
        result_text = re.sub(r'<think>[\s\S]*?</think>', '', result_text).strip()
        json_match = re.search(r'\[[\s\S]*\]', result_text)
        if not json_match:
            print(f"  [调试] 未找到JSON，模型输出: {result_text[:200]}")
            return None
        try:
            return json.loads(json_match.group())
        except Exception:
            return repair_json(result_text)
    except Exception as e:
        print(f"  模型调用异常: {e}")
        return None


def batch_analyze_comments(comments_with_idx, max_retries=3):
    """批量分析评论。
    若模型漏输出某些条目（小模型批量时常见），对缺失条目逐条单独重试。
    根本原因：同批含超长评论时，小模型注意力偏移，漏输出短评论。
    """
    for attempt in range(max_retries):
        raw = _call_model_once(comments_with_idx)
        if raw:
            results = post_process_results(raw)
            # 检测漏输出的条目
            returned_idxs = {r.get('序号') for r in results}
            missing = [(idx, c) for idx, c in comments_with_idx if idx not in returned_idxs]
            if missing:
                print(f"  [补漏] 批次漏输出 {len(missing)} 条，逐条单独重试...")
                for item in missing:
                    # 单条重试2次，仍失败则用占位记录
                    single_raw = None
                    for _ in range(2):
                        single_raw = _call_model_once([item])
                        if single_raw:
                            break
                        time.sleep(1)
                    if single_raw:
                        results.extend(post_process_results(single_raw))
                    else:
                        results.append(make_fail_record(item[0]))
            return results

        print(f"  批次分析失败 (尝试 {attempt+1}/{max_retries})")
        time.sleep(2)

    return [make_fail_record(idx) for idx, _ in comments_with_idx]


def _build_cols():
    """动态构建输出列顺序"""
    base = ['序号', '昵称', '评论内容', '来源', '车型', '评论时间', '点赞数',
            '有效无效', '正负面', '营销手段', '技术亮点']
    suffix = TECH_DIMS + TECH_DIM_DESCS + ['技术点摘要']
    return base + suffix


def save_incremental_csv(df_original, batch_results, source_name="全部"):
    """将新的一批分析结果与原始数据合并后追加到CSV"""
    if not batch_results:
        return

    df_analysis = pd.DataFrame(batch_results)
    target_indices = [item['序号'] for item in batch_results]
    df_subset = df_original.iloc[target_indices].copy()
    df_subset['序号'] = df_subset.index

    df_merged = pd.merge(df_subset, df_analysis, on='序号', how='left')

    cols = _build_cols()
    if '地区' in df_merged.columns:
        cols.insert(6, '地区')
    final_cols = [c for c in cols if c in df_merged.columns]
    df_final = df_merged[final_cols]

    csv_file = os.path.join(OUTPUT_DIR, f"评论分析结果_{source_name}.csv")
    file_exists = os.path.isfile(csv_file)
    df_final.to_csv(csv_file, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')


def process_parallel(df, total_workers=4, source_name="全部"):
    """并行处理评论"""
    comments = df['评论内容'].tolist()
    total = len(comments)

    print(f"\n开始并行分析，共 {total} 条评论")
    print(f"使用模型: {OLLAMA_MODEL}, 并行线程: {total_workers}, 每批: {BATCH_SIZE} 条, num_ctx: {NUM_CTX}")
    print(f"技术维度: {', '.join(TECH_DIMS)}")
    print("="*60)

    csv_file = os.path.join(OUTPUT_DIR, f"评论分析结果_{source_name}.csv")
    if os.path.exists(csv_file):
        try:
            os.remove(csv_file)
            print(f"已清理旧文件: {csv_file}")
        except Exception as e:
            print(f"警告: 无法清理旧文件: {e}")

    batches = []
    for i in range(0, total, BATCH_SIZE):
        batch = [(i + j, comments[i + j]) for j in range(min(BATCH_SIZE, total - i))]
        batches.append(batch)

    print(f"共分为 {len(batches)} 个批次")

    all_results = []
    completed = 0
    failed_batches = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_to_batch = {executor.submit(batch_analyze_comments, batch): i
                          for i, batch in enumerate(batches)}

        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed += 1
                save_incremental_csv(df, results, source_name)

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(batches) - completed) / rate if rate > 0 else 0
                print(f"进度: {completed}/{len(batches)} 批次 | "
                      f"已保存: {len(all_results)} 条 | "
                      f"预计剩余: {remaining/60:.1f}分钟")

            except Exception as e:
                print(f"批次 {batch_idx} 处理异常: {e}")
                failed_batches.append(batch_idx)

    if failed_batches:
        print(f"\n重试失败的 {len(failed_batches)} 个批次...")
        for batch_idx in failed_batches:
            results = batch_analyze_comments(batches[batch_idx], max_retries=5)
            all_results.extend(results)
            save_incremental_csv(df, results, source_name)

    all_results.sort(key=lambda x: x.get('序号', 0))
    return all_results


def analyze_all_comments(df, source_name="全部"):
    """分析所有评论的主函数"""
    try:
        return process_parallel(df, MAX_WORKERS, source_name)
    except Exception as e:
        print(f"并行处理失败: {e}，改用串行...")
        csv_file = os.path.join(OUTPUT_DIR, f"评论分析结果_{source_name}.csv")
        if os.path.exists(csv_file):
            try: os.remove(csv_file)
            except: pass

        results = []
        comments = df['评论内容'].tolist()
        for i in range(0, len(comments), BATCH_SIZE):
            batch = [(i + j, comments[i + j]) for j in range(min(BATCH_SIZE, len(comments) - i))]
            batch_results = batch_analyze_comments(batch)
            results.extend(batch_results)
            save_incremental_csv(df, batch_results, source_name)
            print(f"进度: {min(i+BATCH_SIZE, len(comments))}/{len(comments)}")
        return results


def save_results(df_original, results, source_name="全部"):
    """保存最终汇总结果（全量备份）"""
    print("\n" + "="*60)
    print("保存最终汇总结果...")
    print("="*60)

    df_results = pd.DataFrame(results)
    df_original = df_original.copy()
    df_original['序号'] = range(len(df_original))

    df_output = pd.merge(df_original, df_results, on='序号', how='left')

    cols = _build_cols()
    if '地区' in df_output.columns:
        cols.insert(6, '地区')
    cols = [c for c in cols if c in df_output.columns]
    df_output = df_output[cols]

    safe_name = source_name.replace(" ", "_").replace("+", "_")

    output_file = os.path.join(OUTPUT_DIR, f"评论分析结果_{safe_name}.xlsx")
    try:
        df_output.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Excel已保存: {output_file}")
    except PermissionError:
        print(f"[警告] Excel文件被占用，跳过Excel保存（CSV已保存）: {output_file}")

    csv_file = os.path.join(OUTPUT_DIR, f"评论分析结果_{safe_name}.csv")
    df_output.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"CSV已保存: {csv_file}")

    return df_output


def generate_statistics(df_results):
    """生成统计报告"""
    print("\n" + "="*60)
    print("[统计] 分析结果统计")
    print("="*60)

    total = len(df_results)
    if total == 0:
        return

    for label, col in [("有效无效", '有效无效'), ("正负面", '正负面'),
                        ("营销手段", '营销手段'), ("技术亮点", '技术亮点')]:
        print(f"\n[{label}]:")
        if col in df_results.columns:
            for k, v in df_results[col].value_counts().items():
                print(f"    {k}: {v} ({v/total*100:.1f}%)")

    print("\n[技术维度涉及率（有效评论）]:")
    if '有效无效' in df_results.columns:
        valid_df = df_results[df_results['有效无效'] == '有效']
        v_total = len(valid_df)
        if v_total > 0:
            for dim in TECH_DIMS:
                if dim in valid_df.columns:
                    pos = (valid_df[dim] == '正面').sum()
                    neg = (valid_df[dim] == '负面').sum()
                    mentioned = pos + neg
                    if mentioned > 0:
                        print(f"    {dim}: 提及{mentioned}条({mentioned/v_total*100:.1f}%) "
                              f"| 正面{pos} 负面{neg}")

    if '有效无效' in df_results.columns and '正负面' in df_results.columns:
        print("\n[有效评论中正负面分布]:")
        valid_df = df_results[df_results['有效无效'] == '有效']
        for k, v in valid_df['正负面'].value_counts().items():
            print(f"    {k}: {v} ({v/len(valid_df)*100:.1f}%)")


def check_ollama():
    """检查ollama状态"""
    global OLLAMA_MODEL
    print("\n" + "="*60)
    print("[检查] Ollama 状态")
    print("="*60)
    try:
        models = ollama.list()
        if hasattr(models, 'get'):
            model_list = models.get('models', [])
        elif hasattr(models, 'models'):
            model_list = models.models
        else:
            model_list = []

        available_models = []
        for m in model_list:
            name = m.get('name', m.get('model', ''))
            if name:
                available_models.append(name)

        if not available_models:
            print(f"[警告] 未找到已下载的模型，请运行: ollama pull {MODEL_NAME}")
            return False

        print(f"[OK] Ollama 服务正常")
        print(f"[OK] 可用模型: {available_models}")

        matched_model = None
        for m in available_models:
            if MODEL_NAME in m or m in MODEL_NAME:
                matched_model = m
                break

        if matched_model:
            OLLAMA_MODEL = matched_model
            print(f"[OK] 匹配到模型: {OLLAMA_MODEL}")
            return True
        else:
            print(f"\n[警告] 未找到模型 '{MODEL_NAME}'，请运行: ollama pull {MODEL_NAME}")
            return False

    except Exception as e:
        print(f"[错误] 无法连接Ollama服务: {e}")
        print("请先启动Ollama: ollama serve")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("[评论分析 Agent]")
    print("="*60)

    dongche_count = 0
    douyin_count = 0
    try:
        f = os.path.join(DATA_DIR, "懂车帝评论.xlsx")
        if os.path.exists(f):
            xlsx = pd.ExcelFile(f)
            for sheet in xlsx.sheet_names:
                dongche_count += len(xlsx.parse(sheet))
        f = os.path.join(DATA_DIR, "抖音评论.xlsx")
        if os.path.exists(f):
            xlsx = pd.ExcelFile(f)
            for sheet in xlsx.sheet_names:
                douyin_count += len(xlsx.parse(sheet))
    except Exception:
        pass

    total_count = dongche_count + douyin_count
    print(f"[配置] 模型: {MODEL_NAME} | 硬件: RTX 5070 12GB")
    print(f"       并发: {MAX_WORKERS} 线程 | 批次: {BATCH_SIZE} 条/批 | num_ctx: {NUM_CTX}")
    print(f"[数据] 懂车帝: {dongche_count} 条 | 抖音: {douyin_count} 条 | 合计: {total_count} 条")
    print("="*60)

    print("\n[选择] 请选择处理数据源:")
    print("   1. 懂车帝")
    print("   2. 抖音")
    print("   3. 全部")
    print("   0. 退出")

    source_choice = input("\n请输入选项 (0-3): ").strip()
    source_map = {'1': ('dongche', dongche_count), '2': ('douyin', douyin_count), '3': (None, total_count)}

    if source_choice == '0':
        print("已退出"); return
    elif source_choice not in source_map:
        print("无效选项"); return

    source, source_total = source_map[source_choice]
    source_name = {'dongche': '懂车帝', 'douyin': '抖音', None: '全部'}[source]

    print(f"\n[选择] {source_name}数据量:")
    print("   1. 测试 10 条")
    print("   2. 测试 50 条")
    print("   3. 测试 100 条")
    print("   4. 测试 500 条")
    print("   5. 测试 1000 条")
    print(f"   6. 全部 {source_total} 条")

    limit_choice = input("\n请输入选项 (1-6): ").strip()
    limit_options = {'1': 10, '2': 50, '3': 100, '4': 500, '5': 1000, '6': source_total}

    global MAX_COMMENTS
    MAX_COMMENTS = limit_options.get(limit_choice, source_total)
    print(f"\n[确认] 将处理 {source_name} 数据: {MAX_COMMENTS} 条")
    print("="*60)

    if not check_ollama():
        print("\n[错误] 请先启动Ollama服务"); return

    df = load_data(limit=MAX_COMMENTS, source=source)
    if df.empty:
        return

    results = analyze_all_comments(df, source_name=source_name)
    df_output = save_results(df, results, source_name=source_name)
    generate_statistics(df_output)

    print("\n" + "="*60)
    print("[完成] 分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
