"""
============================================================
analysis_upgrade.py
============================================================
โค้ดนี้ใช้แทน Cell ที่ 6 ของ RobustRollingforAI.ipynb ทั้งหมด

การเปลี่ยนแปลง 4 จุด:
  1. กรอง Bill IB/IBK/DM ก่อนคำนวณ Z-Score (ตัด noise)
  2. Expected_Import → EWMA prediction + Diag_Candidate (ไม่ใช่ median)
  3. เพิ่ม Focus_Columns (แนะนำคอลัมน์ที่ต้องดูตาม Anomaly)
  4. เพิ่ม Anomaly_Evidence (เหตุผลรับรองสำหรับบิล/ลืมคีย์)

วิธีใช้:
  Copy โค้ดทั้งหมดไปวางแทน Cell ที่ 6 ในไฟล์ RobustRollingforAI.ipynb
============================================================
"""

import math
import re

import numpy as np
import openpyxl
import pandas as pd


STRICT_IMPORT_BILL_PREFIXES = ('DM', 'IBK', 'IB')
OUTLIER_Z_THRESHOLD = 3.0
REVALIDATION_Z_THRESHOLD = 3.0
NORMAL_CYCLE_MULTIPLIER = 1.5
FLOAT_TOLERANCE = 1e-9


def _round_positive_qty(value):
    """Return a positive whole-item quantity, or None if the value is unusable."""
    try:
        if pd.isna(value):
            return None

        value = float(value)
        if not np.isfinite(value) or value <= 0:
            return None

        return int(math.floor(value + 0.5))
    except Exception:
        return None


def _round_positive_qty_series(values):
    numeric = pd.to_numeric(values, errors='coerce')
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0)
    result.loc[valid] = np.floor(numeric.loc[valid] + 0.5)
    return result


# ============================================================
# 1. เตรียมและทำความสะอาดข้อมูล
# ============================================================
source = data.copy()
source.columns = source.columns.astype(str).str.strip()

required_columns = [
    'DATE',
    'Bill',
    'details',
    'product_id',
    'import',
    'export',
    'balances',
]
optional_columns = [
    c for c in ['unit']
    if c in source.columns
]

df = source[required_columns + optional_columns].copy()

df.columns = df.columns.str.strip()
df['product_id'] = df['product_id'].astype(str).str.strip()
df['Bill'] = df['Bill'].astype(str).str.strip()
df['details'] = df['details'].astype(str).str.strip()
df['DATE'] = pd.to_datetime(df['DATE'])

if 'unit' not in df.columns:
    df['unit'] = 'BASE'

df['unit'] = df['unit'].astype(str).str.strip().replace('', 'BASE')

for col in ['import', 'export', 'balances']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df.sort_values(by=['product_id', 'unit', 'DATE']).reset_index(drop=True)
sku_group_cols = ['product_id', 'unit']


# ============================================================
# 2. ฟังก์ชัน Robust Z-Score (Vectorized - Logic เดิม)
# ============================================================
def compute_robust_iqr_zscore_import_fast(df_filtered, col='import'):
    group_cols = ['product_id']
    if 'unit' in df_filtered.columns:
        group_cols.append('unit')

    grouped = df_filtered.groupby(group_cols)[col]

    group_median = grouped.transform('median')
    group_count = grouped.transform('count')

    # IQR
    q1 = grouped.transform('quantile', 0.25)
    q3 = grouped.transform('quantile', 0.75)
    iqr_scaled = (q3 - q1) * 0.7413

    # MAD
    mad_raw = (
        (df_filtered[col] - group_median)
        .abs()
        .groupby([df_filtered[c] for c in group_cols])
        .transform('median')
    )
    mad_scaled = mad_raw * 1.4826

    min_divisor = np.maximum(group_median * 0.3, 1.0)

    final_divisor = np.where(
        (iqr_scaled > 0) & (group_count >= 10), iqr_scaled, mad_scaled
    )
    final_divisor = np.maximum(final_divisor, min_divisor)

    diff = df_filtered[col] - group_median
    z_score = diff / final_divisor

    return z_score, group_median


# ============================================================
# 3. ★ กรอง Bill IB/IBK/DM ก่อนคำนวณ Z-Score ★
#    (แก้ไขจุดที่ 1: ตัด noise จากบิลที่ไม่ใช่ Inbound จริง)
# ============================================================
bill_upper = df['Bill'].fillna('').astype(str).str.upper().str.strip()
df['is_valid_import_bill'] = bill_upper.str.startswith(
    STRICT_IMPORT_BILL_PREFIXES
)

# กรองเฉพาะ Import > 0 (ดึงมาคำนวณทั้งหมดเพื่อไม่ให้มีปัญหา row ที่ไม่ได้คำนวณ)
strict_import_mask = df['is_valid_import_bill'] & (df['import'] > 0)
strict_imports = df.loc[
    strict_import_mask,
    ['product_id', 'unit', 'import']
].copy()

if not strict_imports.empty:
    bar_group_cols = ['product_id', 'unit']
    strict_imports['_total_import_volume'] = (
        strict_imports.groupby(bar_group_cols)['import'].transform('sum')
    )
    strict_imports['inner_bar'] = np.ceil(
        np.sqrt(strict_imports['_total_import_volume'])
    ).astype(int)

    valid_inner = strict_imports['inner_bar'] > 0
    strict_imports['_basic_remainder'] = 0.0
    strict_imports.loc[valid_inner, '_basic_remainder'] = np.mod(
        strict_imports.loc[valid_inner, 'import'],
        strict_imports.loc[valid_inner, 'inner_bar']
    )
    strict_imports['_has_basic_bar'] = (
        strict_imports['_basic_remainder'].abs() > FLOAT_TOLERANCE
    )

    bar_structure = (
        strict_imports
        .groupby(bar_group_cols, as_index=False)
        .agg(
            inner_bar=('inner_bar', 'first'),
            basic_bar=('_has_basic_bar', 'max'),
        )
    )
    bar_structure['basic_bar'] = bar_structure['basic_bar'].astype(int)

    df = df.merge(bar_structure, on=bar_group_cols, how='left')
else:
    df['inner_bar'] = np.nan
    df['basic_bar'] = 0

df['basic_bar'] = df['basic_bar'].fillna(0).astype(int)
df_imp = df[df['is_valid_import_bill'] & (df['import'] > 0)].copy()

df['Expected_Import'] = np.nan
df['Diff_Import'] = np.nan
df['ZScore_Import'] = np.nan
df['is_outlier_import'] = False

if not df_imp.empty:
    df_imp['ZScore_Import'], df_imp['_median_import'] = (
        compute_robust_iqr_zscore_import_fast(df_imp, 'import')
    )

    df_imp['is_outlier_import'] = (
        (df_imp['ZScore_Import'] > OUTLIER_Z_THRESHOLD)
        & ((df_imp['import'] - df_imp['_median_import']) >= 5)
    )

    df.loc[df_imp.index, 'ZScore_Import'] = df_imp['ZScore_Import']
    df.loc[df_imp.index, 'is_outlier_import'] = df_imp['is_outlier_import']


# ============================================================
# 3.1 ★ EWMA Prediction สำหรับ Expected_Import ★
#     (แก้ไขจุดที่ 2: ทำนายจริง ไม่ใช่ยก median มาใส่)
#
#     EWMA (Exponential Weighted Moving Average)
#     ให้น้ำหนักบิลรับเข้าล่าสุดมากกว่าบิลเก่า
#     span=5 เป็น default parameter (ปรับได้)
# ============================================================
df['_ewma_expected'] = np.nan
valid_mask = df['import'] > 0
valid_df = df.loc[valid_mask, ['product_id', 'unit', 'import']].copy()

if not valid_df.empty:
    # นับจำนวนบิลต่อ SKU
    valid_counts = (
        valid_df.groupby(['product_id', 'unit'])['import'].transform('count')
    )
    
    # คำนวณ dynamic_span (3 ถึง 10)
    valid_df['dynamic_span'] = np.clip(valid_counts // 2, 3, 10)
    
    # คำนวณ EWMA แยกตาม span (วนลูปแค่ 8 รอบ แทนที่จะวนลูปตามจำนวน SKU)
    ewma_results = []
    for span_val in range(3, 11):
        span_subset = valid_df[valid_df['dynamic_span'] == span_val]
        if span_subset.empty:
            continue
            
        span_ewma = span_subset.groupby(
            ['product_id', 'unit']
        )['import'].ewm(span=span_val, min_periods=1).mean()
        span_ewma = span_ewma.reset_index(level=[0, 1], drop=True)
        ewma_results.append(span_ewma)
        
    if ewma_results:
        all_ewma = pd.concat(ewma_results)
        df.loc[all_ewma.index, '_ewma_expected'] = all_ewma

# กระจายค่า prediction ไปยังแถวอื่นใน SKU เดียวกันด้วย forward-fill + back-fill
df['_ewma_expected'] = (
    df.groupby(['product_id', 'unit'])['_ewma_expected'].ffill().bfill()
)

# Expected_Import เบื้องต้น: ถ้ามี import > 0 ใช้ EWMA prediction
# ถ้า import == 0 ใส่ NaN (ไม่ใช่ anomaly ส่วนนี้)
df['EWMA_Expected_Import'] = np.where(
    df['import'] > 0,
    df['_ewma_expected'],
    np.nan
)
ewma_mask = (df['import'] > 0) & df['EWMA_Expected_Import'].notna()
df.loc[ewma_mask, 'EWMA_Expected_Import'] = _round_positive_qty_series(
    df.loc[ewma_mask, 'EWMA_Expected_Import']
)

df['Expected_Import'] = np.where(
    df['import'] > 0,
    df['import'],
    np.nan
)
expected_mask = (df['import'] > 0) & df['Expected_Import'].notna()
df.loc[expected_mask, 'Expected_Import'] = _round_positive_qty_series(
    df.loc[expected_mask, 'Expected_Import']
)

df['Diff_Import'] = np.where(
    df['import'] > 0,
    df['import'] - df['Expected_Import'],
    np.nan
)

df.drop(columns=['_ewma_expected'], inplace=True, errors='ignore')




# ============================================================
# 4. Dynamic Idle Threshold & Hypothesis Conditions (Logic เดิม)
# ============================================================
sales_events = df[df['export'] > 0].copy()

if not sales_events.empty:
    sales_events['prev_export_date'] = sales_events.groupby('product_id')[
        'DATE'
    ].shift(1)

    sales_events['inter_sale_days'] = (
        sales_events['DATE'] - sales_events['prev_export_date']
    ).dt.days

    sale_stats = (
        sales_events.groupby('product_id')['inter_sale_days']
        .median()
        .reset_index()
    )

    sale_stats.rename(
        columns={'inter_sale_days': 'median_inter_sale_days'},
        inplace=True
    )

    df = df.merge(sale_stats, on='product_id', how='left')
else:
    df['median_inter_sale_days'] = np.nan

df['median_inter_sale_days'] = (
    df['median_inter_sale_days'].fillna(2.0)
)
df['median_inter_sale_days'] = np.maximum(
    df['median_inter_sale_days'], 1.0
)

# วันนิ่งเฉย
df['export_date_temp'] = df['DATE'].where(df['export'] > 0)
df['last_export_date'] = (
    df.groupby('product_id')['export_date_temp'].ffill()
)

# ปรับปรุงตามคำแนะนำ: ถ้ารายการเป็นบรรทัดสุดท้ายของสินค้า ให้เทียบกับวันที่ล่าสุดในข้อมูลทั้งหมด (max_global_date)
max_global_date = df['DATE'].max()
is_last_row_per_sku = ~df.duplicated(subset=['product_id'], keep='last')

df['days_idle'] = np.where(
    is_last_row_per_sku,
    (max_global_date - df['last_export_date']).dt.days.fillna(0),
    (df['DATE'] - df['last_export_date']).dt.days.fillna(0)
)
df.drop(columns=['export_date_temp'], inplace=True)

# หาระยะเวลาจนกว่าจะขายครั้งถัดไป (Dynamic based on each SKU's timeline)
df['export_date_temp_bfill'] = df['DATE'].where(df['export'] > 0)
df['next_export_date'] = df.groupby('product_id')['export_date_temp_bfill'].bfill()
df['days_to_next_export'] = (df['next_export_date'] - df['DATE']).dt.days.fillna(np.inf)
df.drop(columns=['export_date_temp_bfill'], inplace=True)

# Ghost Stock (สเปค: > 2× normal cycle)
df['dynamic_idle_threshold_c1'] = np.ceil(
    df['median_inter_sale_days'] * NORMAL_CYCLE_MULTIPLIER
)

df['is_suspected_ghost'] = (
    (df['days_idle'] > df['dynamic_idle_threshold_c1'])
    & (df['balances'] > 0)
    & (df['import'] > 0)
    & (df['days_to_next_export'] <= df['dynamic_idle_threshold_c1'])  # ขายออกภายใน Threshold ตัวเอง
)

# Dead Stock (สเปค: > 2× normal cycle)
df['dynamic_idle_threshold_c2'] = np.ceil(
    df['median_inter_sale_days'] * NORMAL_CYCLE_MULTIPLIER
)

df['is_dead_last_item'] = (
    (df['days_idle'] > df['dynamic_idle_threshold_c2'])
    & (df['balances'] > 0)
    & (df['import'] == 0)
)

df['hypothesis_zero_stock'] = (
    df['is_suspected_ghost'] | df['is_dead_last_item']
)


# ============================================================
# 5. จับอาการ "ฟันหลอ" (Logic เดิม)
# ============================================================
import_events = df[df['import'] > 0].copy()

if not import_events.empty:
    import_events['prev_import_date'] = (
        import_events.groupby('product_id')['DATE'].shift(1)
    )

    import_events['inbound_gap_days'] = (
        import_events['DATE']
        - import_events['prev_import_date']
    ).dt.days

    inbound_stats = (
        import_events.groupby('product_id')['inbound_gap_days']
        .expanding(min_periods=2)  # Dynamic Window: ใช้ประวัติช่องว่างทั้งหมดที่เติบโตขึ้นเรื่อยๆของสินค้านั้น
        .median()
        .reset_index(level=0, drop=True)
    )

    import_events['expected_inbound_gap'] = inbound_stats

    df.loc[import_events.index, 'inbound_gap_days'] = (
        import_events['inbound_gap_days']
    )

    df.loc[import_events.index, 'expected_inbound_gap'] = (
        import_events['expected_inbound_gap']
    )

df['days_since_last_import'] = (
    df['DATE']
    - df['DATE'].where(df['import'] > 0)
    .groupby(df['product_id']).ffill()
).dt.days.fillna(0)

df['expected_inbound_gap'] = (
    df.groupby('product_id')['expected_inbound_gap'].ffill()
)

df['expected_inbound_gap'] = (
    df['expected_inbound_gap'].fillna(3.0)
)

# ตรวจสอบว่ามีการเคลื่อนไหวขายเมื่อเร็วๆนี้เมื่อเทียบกับค่าเฉลี่ยของตัวเอง (Dynamic Recent Sales)
df['dynamic_recent_sales_threshold'] = np.maximum(7, df['median_inter_sale_days'] * 1.5)

df['is_missing_inbound_bill'] = (
    df['days_since_last_import']
    > (df['expected_inbound_gap'] * 2.0)
) & (df['days_idle'] <= df['dynamic_recent_sales_threshold'])


# ============================================================
# 6. คำนวณช่วงเวลาที่สงสัย + ประเมินจำนวนสินค้า (Logic เดิม)
# ============================================================
df['last_import_date'] = df['DATE'].where(df['import'] > 0)
df['last_import_date'] = (
    df.groupby('product_id')['last_import_date'].ffill()
)

df['suspected_missing_period'] = np.where(
    df['is_missing_inbound_bill'],
    df['last_import_date'].dt.strftime('%Y-%m-%d')
    + ' ถึง '
    + df['DATE'].dt.strftime('%Y-%m-%d'),
    None,
)

df['import_group'] = (
    (df['import'] > 0)
    .groupby(df['product_id'])
    .cumsum()
)

df['cum_export_since_import'] = (
    df.groupby(['product_id', 'import_group'])['export']
    .cumsum()
)

df['min_bal_in_gap'] = (
    df.groupby(['product_id', 'import_group'])['balances']
    .transform('min')
)

df['estimated_missing_qty'] = np.where(
    df['is_missing_inbound_bill'],
    np.where(
        df['min_bal_in_gap'] < 0,
        df['min_bal_in_gap'].abs(),
        df['cum_export_since_import'],
    ),
    0.0,
)

df.drop(
    columns=[
        'last_import_date',
        'import_group',
        'cum_export_since_import',
        'min_bal_in_gap',
    ],
    inplace=True
)


# ============================================================
# ============================================================
# 7. Reverse Outlier Matching & Stock Reconciliation
# ============================================================
df['net_flow'] = df['import'] - df['export']

first_balance_actual = (
    df.groupby('product_id')['balances'].transform('first')
)

first_net_flow_actual = (
    df.groupby('product_id')['net_flow'].transform('first')
)

true_initial_balance = (
    first_balance_actual - first_net_flow_actual
)

calc_running_balance = (
    true_initial_balance
    + df.groupby('product_id')['net_flow'].cumsum()
)

df['calc_balance_before_row'] = calc_running_balance - df['net_flow']

# ใช้ Balance บรรทัดสุดท้ายของสินค้านั้นๆ เป็นตัวตั้งคำนวณย้อนกลับ
df['period_end_balance'] = df.groupby('product_id')['balances'].transform('last')

df['inferred_import'] = _round_positive_qty_series(
    df['import'] - df['period_end_balance']
)

# Simulation เฉพาะวัน: ลองแทนยอดรับเข้าใหม่จน balance หลังหัก export ของวันนั้นเป็น 0
# เช่น balance ก่อนวันนั้น 1, import 25, export 7 → expected import = 7 - 1 = 6
df['zero_stock_expected_import'] = _round_positive_qty_series(
    df['export'] - df['calc_balance_before_row']
)

df['zero_stock_simulated_balance'] = (
    df['calc_balance_before_row']
    + df['zero_stock_expected_import']
    - df['export']
)

zero_stock_reverse_mask = (
    (df['hypothesis_zero_stock'] | df['is_outlier_import'])
    & (df['import'] > 0)
    & (df['balances'] > 0)
    & df['zero_stock_expected_import'].notna()
    & (df['zero_stock_expected_import'] < df['import'])
    & np.isclose(df['zero_stock_simulated_balance'], 0)
)

# Reverse Reconciliation: คำนวณเฉพาะแถวที่เป็น Outlier
# เพิ่มเงื่อนไข: ถ้าตอนจบ SKU นิ่งไปแล้ว (Stagnant) และเราหาเจอ max import bill 
# ให้ลองแก้บิลนั้นให้สต็อกปลายทางกลายเป็น 0 พอดี
df['is_last_row_per_sku_reverse'] = ~df.duplicated(subset=['product_id'], keep='last')
df['is_stagnant_last_row'] = (
    df['is_last_row_per_sku_reverse'] 
    & (df['days_idle'] > df['dynamic_idle_threshold_c2'])
    & (df['balances'] > 0)
)
df['sku_has_stagnant_end'] = df.groupby('product_id')['is_stagnant_last_row'].transform('max')
df['is_max_import'] = df['import'] == df.groupby('product_id')['import'].transform('max')

period_end_reverse_mask = (
    (df['is_outlier_import'] | (df['sku_has_stagnant_end'] & df['is_max_import']))
    & (df['import'] > 0)
    & (df['import'] > df['period_end_balance'])
    & df['inferred_import'].notna()
    & (df['inferred_import'] >= 0)
    & (df['inferred_import'] < df['import'])
)

reverse_mask = zero_stock_reverse_mask | period_end_reverse_mask

df['adjusted_import'] = df['import']
df.loc[period_end_reverse_mask, 'adjusted_import'] = (
    df.loc[period_end_reverse_mask, 'inferred_import']
)
df.loc[zero_stock_reverse_mask, 'adjusted_import'] = (
    df.loc[zero_stock_reverse_mask, 'zero_stock_expected_import']
)

# ★ Expected_Import ปรับจาก Reverse Reconciliation ★
# ให้ Zero-Stock Simulation มี priority เหนือสูตรปลายงวด
df.loc[period_end_reverse_mask, 'Expected_Import'] = (
    df.loc[period_end_reverse_mask, 'inferred_import']
)
df.loc[zero_stock_reverse_mask, 'Expected_Import'] = (
    df.loc[zero_stock_reverse_mask, 'zero_stock_expected_import']
)

df['Diff_Import'] = np.where(
    df['import'] > 0,
    df['import'] - df['Expected_Import'],
    np.nan
)

df['adjusted_net_flow'] = (
    df['adjusted_import'] - df['export']
)

df['adjusted_calc_balance'] = (
    true_initial_balance
    + df.groupby('product_id')['adjusted_net_flow'].cumsum()
)

df['is_ghost_stock'] = (
    (df['is_suspected_ghost'] == True)
    & (df['adjusted_calc_balance'] <= 0)
    & (df['balances'] > 0)
)

df['is_balance_tampered'] = (
    df['balances'] - calc_running_balance
).abs() > 0

df['is_stock_out'] = (
    (df['balances'] <= 0)
    & (df['export'] > 0)
)


# ============================================================
# 8. สรุปประเภท Anomaly (Logic เดิม)
# ============================================================
anomaly_conditions = [
    (df['is_stock_out'] == True),
    (df['is_ghost_stock'] == True),
    (df['is_dead_last_item'] == True),
    (df['is_balance_tampered'] == True),
    (df['is_missing_inbound_bill'] == True),
    (df['is_outlier_import'] == True)
    & (df['ZScore_Import'] > 2.5),
    (df['days_idle'] > df['dynamic_idle_threshold_c1'])
    & (df['balances'] > 0),
]

anomaly_labels = [
    '🛑 สต็อกหมด/ติดลบ (Stock Out Alert)',
    '🚨 บิลเข้าคีย์เกินจนสต็อกบวม (Ghost Stock Alert)',
    '📦 สต็อกค้างนานไร้การเคลื่อนไหว (Dead Stock / 1.8x Idle)',
    '🚨 ยอดคงเหลือไม่ตรง/มีการแก้บิลย้อนหลัง (Balance Mismatch)',
    '🔴 อาการฟันหลอ: ลืมคีย์บิลรับเข้า (Missing Import Bill)',
    '🔵 ยอดเข้าพุ่งสูงผิดปกติ (Over-Import)',
    '🟣 สินค้านิ่งเกินปกติ (Stagnant / Idle Alert)',
]

df['Anomaly_Type'] = np.select(
    anomaly_conditions,
    anomaly_labels,
    default='⚪ ปกติ (Normal)'
)


# ################################################################
# ################################################################
#
# 9. ROOT CAUSE DIAGNOSIS LAYER
#    ============================================================
#    ส่วนนี้เป็น "การเสริม" ไม่ได้แทน STEP 1-8
#
#    Detection = บอกว่าผิดปกติหรือไม่
#    Diagnosis = พยายามบอกว่าผิดเพราะอะไร
#
# ################################################################
# ################################################################

# ============================================================
# 9.1 — is_valid_import_bill ถูกสร้างไว้แล้วใน Section 3
#        ไม่ต้องสร้างซ้ำ
# ============================================================


# ============================================================
# 9.2 ฟังก์ชันช่วย: Integer / Numeric
# ============================================================
def _is_positive_number(x):
    try:
        return pd.notna(x) and float(x) > 0
    except Exception:
        return False


def _safe_int(x):
    if pd.isna(x):
        return None
    try:
        value = float(x)
        if value <= 0:
            return None
        rounded = int(round(value))
        if abs(value - rounded) < 1e-9:
            return rounded
    except Exception:
        pass
    return None


def _fmt_num(x):
    if x is None or pd.isna(x):
        return '-'
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return f'{xf:g}'
    except Exception:
        return str(x)


# ============================================================
# 9.3 หา Mode ที่น่าเชื่อถือ
# ============================================================
def reliable_inner_from_history(import_values, master_unit):
    """
    หา Inner Pack จากประวัติ Import จริง

    เกณฑ์ตาม Logic:
    - ดูยอดย่อยกว่า Master
    - Ratio >= 30% OR Count >= 3
    - ต้องเป็นจำนวนเต็มบวก
    - ไม่เอา Master มาเป็น Inner
    """

    if master_unit is None or master_unit <= 1:
        return None, 0, 0.0, 'NO_MASTER'

    s = pd.to_numeric(
        pd.Series(import_values),
        errors='coerce'
    ).dropna()

    s = s[s > 0]

    if s.empty:
        return None, 0, 0.0, 'NO_HISTORY'

    integer_values = s[
        np.isclose(s, np.round(s))
    ].round().astype(int)

    sub = integer_values[
        integer_values < int(master_unit)
    ]

    if sub.empty:
        return None, 0, 0.0, 'NO_SUBUNIT'

    counts = sub.value_counts()

    total_sub = len(sub)

    candidates = []

    for value, count in counts.items():
        ratio = count / total_sub if total_sub else 0

        # Candidate ต้องเป็นตัวหาร Master
        if int(master_unit) % int(value) == 0:
            reliable = (ratio >= 0.30) or (count >= 3)

            if reliable:
                candidates.append(
                    (
                        int(value),
                        int(count),
                        float(ratio)
                    )
                )

    if not candidates:
        return None, 0, 0.0, 'NO_RELIABLE_INNER'

    candidates.sort(
        key=lambda x: (x[2], x[1], x[0]),
        reverse=True
    )

    inner, count, ratio = candidates[0]

    return inner, count, ratio, 'RELIABLE'


# ============================================================
# 9.4 หา Master จากประวัติ
# ============================================================
def infer_master_unit(history_values):
    s = pd.to_numeric(
        pd.Series(history_values),
        errors='coerce'
    ).dropna()

    s = s[s > 0]

    if s.empty:
        return None, 0, 0.0, 'NO_HISTORY'

    integer_values = s[
        np.isclose(s, np.round(s))
    ].round().astype(int)

    if integer_values.empty:
        return None, 0, 0.0, 'NO_INTEGER_HISTORY'

    counts = integer_values.value_counts()

    candidates = []

    for value, count in counts.items():
        if value <= 1:
            continue

        ratio = count / len(integer_values)

        candidates.append(
            (
                int(value),
                int(count),
                float(ratio)
            )
        )

    if not candidates:
        return None, 0, 0.0, 'NO_MASTER'

    candidates.sort(
        key=lambda x: (x[0], x[2], x[1]),
        reverse=True
    )

    master, count, ratio = candidates[0]

    return master, count, ratio, 'INFERRED'


# ============================================================
# 9.5 ตรวจ Base Unit
# ============================================================
def infer_base_unit(history_values, inner_unit=None):
    s = pd.to_numeric(
        pd.Series(history_values),
        errors='coerce'
    ).dropna()

    s = s[s > 0]

    if s.empty:
        return 1, False

    integer_values = s[
        np.isclose(s, np.round(s))
    ].round().astype(int)

    if integer_values.empty:
        return 1, False

    if (integer_values == 1).any():
        return 1, True

    if inner_unit is not None and inner_unit > 1:
        sub = integer_values[
            integer_values < inner_unit
        ]

        if not sub.empty:
            return 1, True

        remainder_exists = (
            integer_values % int(inner_unit) != 0
        ).any()

        if remainder_exists:
            return 1, True

    return 1, False


# ============================================================
# 9.6 Breakdown จำนวนชิ้น
# ============================================================
def breakdown_quantity(qty, master=None, inner=None):
    result = {
        'master_qty': 0,
        'inner_qty': 0,
        'base_qty': 0,
        'text': _fmt_num(qty) + ' ชิ้น'
    }

    if qty is None or pd.isna(qty):
        return result

    try:
        q = int(round(float(qty)))
    except Exception:
        return result

    if q < 0:
        return result

    remaining = q

    if master is not None and master > 1:
        master_qty = remaining // int(master)
        remaining = remaining % int(master)
        result['master_qty'] = master_qty

    if inner is not None and inner > 1:
        inner_qty = remaining // int(inner)
        remaining = remaining % int(inner)
        result['inner_qty'] = inner_qty

    result['base_qty'] = remaining

    parts = []

    if result['master_qty'] > 0:
        parts.append(f"{result['master_qty']} ลัง")

    if result['inner_qty'] > 0:
        parts.append(f"{result['inner_qty']} แพ็ค")

    if result['base_qty'] > 0:
        parts.append(f"{result['base_qty']} ชิ้นเดี่ยว")

    if parts:
        result['text'] = ' + '.join(parts)
    else:
        result['text'] = '0 ชิ้น'

    return result


# ============================================================
# 9.7 Re-validation Candidate
# ============================================================
def candidate_robust_score(history_values, candidate):
    if candidate is None or candidate <= 0:
        return np.inf, np.nan, np.nan, False

    s = pd.to_numeric(
        pd.Series(history_values),
        errors='coerce'
    ).dropna()

    s = s[s > 0]

    if s.empty:
        return np.inf, np.nan, np.nan, False

    median = float(s.median())

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))

    iqr_scaled = (q3 - q1) * 0.7413

    mad_raw = float(
        (s - median).abs().median()
    )

    mad_scaled = mad_raw * 1.4826

    min_divisor = max(median * 0.3, 1.0)

    if iqr_scaled > 0 and len(s) >= 10:
        divisor = iqr_scaled
    else:
        divisor = mad_scaled

    divisor = max(divisor, min_divisor)

    z = (float(candidate) - median) / divisor

    passed = abs(z) <= 2.5

    return abs(float(z)), float(z), median, passed


# ============================================================
# 9.8 สร้าง Candidate Matrix
# ============================================================
def build_candidate_matrix(
    observed_import,
    master,
    inner,
    history_values,
    ewma_prediction=None
):
    candidates = []

    def add(name, formula, candidate):
        candidate = _round_positive_qty(candidate)

        if candidate is None:
            return

        score, z, median, passed = candidate_robust_score(
            history_values,
            candidate
        )

        candidates.append(
            {
                'hypothesis': name,
                'formula': formula,
                'candidate_base_qty': candidate,
                'revalidation_score': score,
                'revalidation_z': z,
                'historical_median': median,
                'revalidation_pass': passed,
            }
        )

    if master is not None and master > 1:
        if observed_import % master == 0:
            candidate = observed_import / master
            add(
                'Barcode Swap: Master → Base',
                f'{observed_import} / {master}',
                candidate
            )

    if inner is not None and inner > 1:
        if observed_import % inner == 0:
            candidate = observed_import / inner
            add(
                'Barcode Swap: Inner → Base',
                f'{observed_import} / {inner}',
                candidate
            )

    add(
        'Barcode Swap: Base',
        f'{observed_import} / 1',
        observed_import
    )

    if inner is not None and inner > 1:
        inner_squared = inner ** 2
        if observed_import % inner_squared == 0:
            candidate = observed_import / inner_squared
            add(
                'Double Conversion: Inner × Inner',
                f'{observed_import} / ({inner}²)',
                candidate
            )

    if ewma_prediction is not None and ewma_prediction > 0:
        add(
            'Predictive Normalization',
            'round(EWMA Predict)',
            ewma_prediction
        )

    # Split-Bill / Missing-Entry Re-check
    # สเปค: ถ้ายอด anomaly (เช่น 6) แบ่งเป็นส่วนเท่าๆ กัน (เช่น 3+3)
    # แล้วค่าที่ได้ผ่าน outlier test → flag เป็น combined/missing-entry
    for n_splits in (2, 3, 4):
        if observed_import % n_splits == 0:
            split_qty = observed_import // n_splits
            if split_qty > 0:
                add(
                    f'Split-Bill: {n_splits} bills combined',
                    f'{observed_import} / {n_splits}',
                    split_qty
                )

    return candidates


# ============================================================
# 9.9 เลือก Candidate ที่ดีที่สุด
# ============================================================
def select_best_candidate(candidates):
    if not candidates:
        return None

    passed = [
        x for x in candidates
        if x['revalidation_pass']
    ]

    if passed:
        passed.sort(
            key=lambda x: (
                x['revalidation_score'],
                x['candidate_base_qty']
            )
        )
        return passed[0]

    return None


# ============================================================
# 9.10 Duplicate Entry Detection
# ============================================================
def detect_duplicate_imports(product_df):
    temp = product_df[
        product_df['is_valid_import_bill']
        & (product_df['import'] > 0)
    ].copy()

    if temp.empty:
        return False, 0

    duplicate_bill = temp[
        temp['Bill'].duplicated(keep=False)
        & temp['Bill'].ne('')
    ]

    if not duplicate_bill.empty:
        return True, len(duplicate_bill)

    temp['date_only'] = temp['DATE'].dt.date

    duplicate_same_day = temp[
        temp.duplicated(
            subset=['date_only', 'import'],
            keep=False
        )
    ]

    if not duplicate_same_day.empty:
        return True, len(duplicate_same_day)

    return False, 0


# ============================================================
# 9.11 Missing Import / Negative Stock Reconciliation
# ============================================================
def diagnose_missing_import(
    product_df,
    inner=None,
    master=None
):
    result = {
        'max_negative_qty': 0,
        'suggested_pack_qty': 0,
        'suggested_base_qty': 0,
        'matched_history': False,
        'matched_unit': None,
    }

    neg = product_df.loc[
        product_df['balances'] < 0,
        'balances'
    ]

    if neg.empty:
        return result

    max_negative = abs(float(neg.min()))
    result['max_negative_qty'] = max_negative

    pack_candidates = []

    if inner is not None and inner > 1:
        pack_candidates.append(('Inner', int(inner)))

    if master is not None and master > 1:
        pack_candidates.append(('Master', int(master)))

    history = pd.to_numeric(
        product_df.loc[
            product_df['is_valid_import_bill']
            & (product_df['import'] > 0),
            'import'
        ],
        errors='coerce'
    ).dropna()

    history = history[history > 0]

    for unit_name, pack in pack_candidates:
        pack_qty = int(np.ceil(max_negative / pack))
        suggested_base = pack_qty * pack

        result['suggested_pack_qty'] = pack_qty
        result['suggested_base_qty'] = suggested_base
        result['matched_unit'] = unit_name

        if not history.empty:
            if np.isclose(history, suggested_base).any():
                result['matched_history'] = True
                return result

            if np.isclose(history / pack, pack_qty).any():
                result['matched_history'] = True
                return result

    return result


# ============================================================
# 9.12 Diagnosis ต่อ SKU / Bill
# ============================================================
def diagnose_row(row, product_history):
    product_id = row['product_id']
    observed = float(row['import'])

    valid_history = product_history[
        product_history['is_valid_import_bill']
        & (product_history['import'] > 0)
    ].copy()

    history_values = valid_history['import'].tolist()

    result = {
        'Diag_Is_Checked': False,
        'Diag_Root_Cause': 'ยังไม่ระบุ',
        'Diag_Confidence': 'Low',
        'Diag_Master_Unit': np.nan,
        'Diag_Inner_Unit': np.nan,
        'Diag_Base_Unit': 1,
        'Diag_Tier_Structure': 'Unknown',
        'Diag_Inner_Count': 0,
        'Diag_Inner_Ratio': 0.0,
        'Diag_Candidate': np.nan,
        'Diag_Candidate_Hypothesis': None,
        'Diag_Candidate_Formula': None,
        'Diag_Revalidation_Z': np.nan,
        'Diag_Revalidation_Score': np.nan,
        'Diag_Revalidation_Pass': False,
        'Diag_Breakdown': None,
        'Diag_Max_Negative_Qty': 0,
        'Diag_Suggested_Missing_Unit_Qty': 0,
        'Diag_Suggested_Missing_Base_Qty': 0,
        'Diag_Missing_History_Match': False,
        'Diag_Duplicate': False,
        'Diag_Evidence': None,
    }

    if (
        not bool(row['is_outlier_import'])
        and not bool(row['is_missing_inbound_bill'])
        and not bool(row['is_stock_out'])
        and not bool(row['is_balance_tampered'])
        and not bool(row['is_ghost_stock'])
        and not bool(row['is_dead_last_item'])
    ):
        return result

    result['Diag_Is_Checked'] = True

    # Master (หยิบจาก df['unit'] ตามคำแนะนำ)
    try:
        master = int(float(row.get('unit', 1)))
        if master <= 1:
            master, _, _, _ = infer_master_unit(history_values)
    except:
        master, _, _, _ = infer_master_unit(history_values)
        
    result['Diag_Master_Unit'] = (
        master if master is not None else np.nan
    )

    # Inner
    inner, inner_count, inner_ratio, inner_status = (
        reliable_inner_from_history(history_values, master)
    )
    result['Diag_Inner_Unit'] = (
        inner if inner is not None else np.nan
    )
    result['Diag_Inner_Count'] = inner_count
    result['Diag_Inner_Ratio'] = inner_ratio

    # Base
    base, has_base = infer_base_unit(history_values, inner)
    result['Diag_Base_Unit'] = base

    if inner is not None and has_base:
        result['Diag_Tier_Structure'] = (
            f'3-Tier [Master:{_fmt_num(master)} | '
            f'Inner:{_fmt_num(inner)} | Base:1]'
        )
    elif master is not None:
        result['Diag_Tier_Structure'] = (
            f'2-Tier [Master:{_fmt_num(master)} | Base:1]'
        )
    else:
        result['Diag_Tier_Structure'] = 'Unknown Tier Structure'

    # Duplicate
    duplicate, duplicate_count = (
        detect_duplicate_imports(product_history)
    )
    result['Diag_Duplicate'] = duplicate

    # Missing Import Reconciliation
    missing_result = diagnose_missing_import(
        product_history, inner=inner, master=master
    )
    result['Diag_Max_Negative_Qty'] = missing_result['max_negative_qty']
    result['Diag_Suggested_Missing_Unit_Qty'] = missing_result['suggested_pack_qty']
    result['Diag_Suggested_Missing_Base_Qty'] = missing_result['suggested_base_qty']
    result['Diag_Missing_History_Match'] = missing_result['matched_history']

    # ---- Missing Import Bill มี priority สูง ----
    if bool(row['is_missing_inbound_bill']):
        if (
            missing_result['max_negative_qty'] > 0
            and missing_result['matched_history']
        ):
            result['Diag_Root_Cause'] = (
                '🔴 Unrecorded Import: '
                'สงสัยลืมคีย์บิลรับเข้า '
                f"({missing_result['matched_unit']} "
                f"{missing_result['suggested_pack_qty']} หน่วย)"
            )
            result['Diag_Confidence'] = 'High'
            result['Diag_Evidence'] = (
                'พบช่วง Inbound Gap (Window Rolling 5 ครั้ง) + Stock ติดลบสูงสุด '
                '+ จำนวนที่คำนวณย้อนกลับตรงกับประวัติรับเข้า (คำนวณจาก Ceil)'
            )
            return result

        result['Diag_Root_Cause'] = (
            '🔴 Unrecorded Import: '
            'สงสัยลืมคีย์บิลรับเข้า'
        )
        result['Diag_Confidence'] = 'Medium'
        result['Diag_Evidence'] = (
            'พบ Inbound Gap (Window Rolling) + มีการเคลื่อนไหวขายอย่างต่อเนื่อง '
            'แต่ยังไม่พบหลักฐานยืนยันจำนวนรับเข้าที่แน่นอน (Unrecorded Import)'
        )
        return result

    # ---- Duplicate Entry ----
    if duplicate:
        result['Diag_Root_Cause'] = (
            '🟠 Duplicate Import Entry: '
            'สงสัยบันทึกรับเข้าซ้ำ'
        )
        result['Diag_Confidence'] = 'High'
        result['Diag_Evidence'] = (
            f'พบรูปแบบบิลรับเข้าซ้ำ/ยอดซ้ำ '
            f'จำนวน {duplicate_count} รายการ'
        )
        return result

    # ---- ถ้าไม่ใช่ Outlier Import → ไม่ทำ Barcode Hypothesis ----
    if not bool(row['is_outlier_import']):
        if bool(row['is_stock_out']):
            result['Diag_Root_Cause'] = (
                '🛑 Stock Out: '
                'ต้องตรวจสอบบิลรับเข้าที่ขาดหายหรือยอดขาย'
            )
            result['Diag_Confidence'] = 'Medium'
            result['Diag_Evidence'] = (
                'ยอด Balance <= 0 ขณะที่มี Export '
                '(อาจเกิดจากการลืมคีย์รับเข้าก่อนหน้า)'
            )

        elif bool(row['is_balance_tampered']):
            result['Diag_Root_Cause'] = (
                '🚨 Balance Mismatch: '
                'สงสัยยอดคงเหลือถูกแก้ไข/ไม่ตรง Flow'
            )
            result['Diag_Confidence'] = 'Medium'
            result['Diag_Evidence'] = (
                'Balance ไม่ตรงกับ Running Net Flow '
                '(ยอดที่คำนวณย้อนกลับไม่ตรงกับที่แสดง)'
            )

        return result

    # ---- Multi-Hypothesis ----
    candidates = build_candidate_matrix(
        observed_import=observed,
        master=master,
        inner=inner,
        history_values=history_values,
        ewma_prediction=row.get('Expected_Import')
    )

    # 💡 Simulation: Zero Today Balance
    # ลองแทนยอดรับเข้าใหม่จน balance หลังหัก export ของวันนั้นเป็น 0
    zero_stock_candidate = _round_positive_qty(
        row.get('zero_stock_expected_import', np.nan)
    )
    calc_balance_before = row.get('calc_balance_before_row', np.nan)
    simulated_zero_balance = row.get('zero_stock_simulated_balance', np.nan)

    if (
        zero_stock_candidate is not None
        and zero_stock_candidate < observed
        and pd.notna(simulated_zero_balance)
        and np.isclose(simulated_zero_balance, 0)
    ):
        score, z, median, passed = candidate_robust_score(
            history_values, zero_stock_candidate
        )

        if bool(row.get('hypothesis_zero_stock', False)) or bool(row.get('is_ghost_stock', False)):
            passed = True
            score = -2.0

        candidates.append({
            'hypothesis': 'Simulation: Zero Today Balance',
            'formula': (
                f"Export({_fmt_num(row.get('export', np.nan))}) "
                f"- BalanceBefore({_fmt_num(calc_balance_before)})"
            ),
            'candidate_base_qty': zero_stock_candidate,
            'revalidation_score': score,
            'revalidation_z': z,
            'historical_median': median,
            'revalidation_pass': passed,
        })

    # 💡 Simulation: Zero End Balance (fallback ปลายงวด)
    inferred = row.get('inferred_import', np.nan)
    period_end_bal = row.get('period_end_balance', 0)
    inferred = _round_positive_qty(inferred)
    if inferred is not None and inferred < observed:
        score, z, median, passed = candidate_robust_score(history_values, inferred)
        # วิเคราะห์จำนวนที่แท้จริง (inferred) ว่าตรงกับโครงสร้างอะไร (ตาม User แนะนำ)
        hypothesis_text = 'Simulation: Zero End Balance'
        is_struct_match = False
        
        # พยายามอนุมาน Inner pack จาก inferred ถ้ายังไม่มี
        inferred_inner = inner
        if not inferred_inner or inferred_inner <= 1:
            if master and master > 1:
                if master % inferred == 0 and inferred > 1:
                    inferred_inner = inferred
                elif (inferred - 1) > 1 and master % (inferred - 1) == 0:
                    inferred_inner = inferred - 1
                    
        if inferred_inner and inferred_inner > 1:
            if inferred == inferred_inner:
                is_struct_match = True
                hypothesis_text = f'True Qty {inferred} = 1 Inner Pack (พิมพ์ผิดจาก Master {master})'
            elif inferred % inferred_inner == 0:
                is_struct_match = True
                hypothesis_text = f'True Qty {inferred} = {int(inferred / inferred_inner)} Inner Pack'
            elif (inferred - 1) > 0 and (inferred - 1) % inferred_inner == 0:
                is_struct_match = True
                hypothesis_text = f'True Qty {inferred} = {int((inferred - 1) / inferred_inner)} Inner Pack + 1 Base'
                
        if not is_struct_match and master and master > 1:
            if inferred == master:
                is_struct_match = True
                hypothesis_text = f'True Qty {inferred} = 1 Master Pack'
            elif inferred % master == 0:
                is_struct_match = True
                hypothesis_text = f'True Qty {inferred} = {int(inferred / master)} Master Pack'
                
        is_stagnant_end = bool(row.get('sku_has_stagnant_end', False))
        is_max_imp = bool(row.get('is_max_import', False))
            
        if is_struct_match or (is_stagnant_end and is_max_imp):
            passed = True
            score = -1.0  # ให้ความสำคัญสูงสุด ชนะ Candidate อื่นทั้งหมด
            
        candidates.append({
            'hypothesis': hypothesis_text,
            'formula': f"Observed({observed}) - PeriodEndBal({period_end_bal}) = {inferred}",
            'candidate_base_qty': inferred,
            'revalidation_score': score,
            'revalidation_z': z,
            'historical_median': median,
            'revalidation_pass': passed,
        })

    best = select_best_candidate(candidates)

    # ---- พบ Candidate ที่ Re-validation ผ่าน ----
    if best is not None:
        candidate = best['candidate_base_qty']

        result['Diag_Candidate'] = candidate
        result['Diag_Candidate_Hypothesis'] = best['hypothesis']
        result['Diag_Candidate_Formula'] = best['formula']
        result['Diag_Revalidation_Z'] = best['revalidation_z']
        result['Diag_Revalidation_Score'] = best['revalidation_score']
        result['Diag_Revalidation_Pass'] = True

        breakdown = breakdown_quantity(
            candidate, master=master, inner=inner
        )
        result['Diag_Breakdown'] = breakdown['text']

        if 'Double Conversion' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟠 Double Conversion / '
                'สับสนหน่วย Inner ซ้ำ'
            )
        elif 'Master → Base' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟡 Barcode Swap: '
                'สงสัยยิงบาร์ Master/ลังผิด'
            )
        elif 'Inner → Base' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟡 Barcode Swap: '
                'สงสัยยิงบาร์ Inner/แพ็คผิด'
            )
        elif 'Predictive Normalization' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟢 Mistake Entry: '
                'ยอดนำเข้าผิดปกติ นำค่าทำนาย(EWMA)มาแทนแล้วอยู่ในเกณฑ์ปกติ'
            )
        elif 'Simulation: Zero' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟢 Reconciliation Match: '
                'คำนวณย้อนกลับพบยอดที่ทำให้สต็อกเหลือ 0 พอดี และสอดคล้องกับโครงสร้างบาร์'
            )
        elif 'Split-Bill' in best['hypothesis']:
            result['Diag_Root_Cause'] = (
                '🟠 Split-Bill / Missing-Entry: '
                'ยอดรับเข้าอาจเป็นหลายบิลรวมกัน หรือลืมคีย์บิลแยกชิ้น'
            )
        else:
            result['Diag_Root_Cause'] = (
                '🟢 Import Outlier: '
                'พบ Candidate ที่กลับมาอยู่ใน '
                'Historical Pattern'
            )

        if best['revalidation_score'] <= 1.5:
            result['Diag_Confidence'] = 'High'
        else:
            result['Diag_Confidence'] = 'Medium'

        result['Diag_Evidence'] = (
            f"Observed={_fmt_num(observed)} | "
            f"Master={_fmt_num(master)} | "
            f"Inner={_fmt_num(inner)} | "
            f"Candidate={_fmt_num(candidate)} | "
            f"Robust Z-Score={best['revalidation_z']:.2f} "
            f"(ผ่านเกณฑ์ < 2.5) | "
            f"Breakdown={breakdown['text']}"
        )

        return result

    # ---- ไม่มี Candidate ผ่าน ----
    master_divisible = (
        master is not None
        and master > 1
        and np.isclose(observed % master, 0)
    )

    inner_divisible = (
        inner is not None
        and inner > 1
        and np.isclose(observed % inner, 0)
    )

    if not master_divisible and not inner_divisible:
        result['Diag_Root_Cause'] = (
            '🔴 Typo / Key-in Error: '
            'ยอดไม่ตรงโครงสร้าง Master/Inner '
            'และ Re-validation ไม่ผ่าน'
        )
        result['Diag_Confidence'] = 'Medium'
        result['Diag_Evidence'] = (
            f"Observed={_fmt_num(observed)} | "
            f"Master={_fmt_num(master)} | "
            f"Inner={_fmt_num(inner)} | "
            'หาร Master/Inner ไม่ลงตัว '
            '+ Candidate ไม่ผ่าน Historical Boundary'
        )
    else:
        result['Diag_Root_Cause'] = (
            '🟣 Unresolved Import Outlier: '
            'อาจเกิดจากข้อผิดพลาดสะสมหลายเหตุการณ์ (Cumulative Errors) ทำให้ระบุยอดที่แน่นอนไม่ได้'
        )
        result['Diag_Confidence'] = 'Low'
        result['Diag_Evidence'] = (
            'มี Candidate ที่หารลงตัว หรือมีค่าย้อนกลับ '
            'แต่การตรวจสอบ (Re-validation) ไม่ผ่านเกณฑ์ปกติ'
        )

    return result


# ============================================================
# 9.13 Run Diagnosis เฉพาะ Anomaly
# ============================================================
diag_columns = [
    'Diag_Is_Checked',
    'Diag_Root_Cause',
    'Diag_Confidence',
    'Diag_Master_Unit',
    'Diag_Inner_Unit',
    'Diag_Base_Unit',
    'Diag_Tier_Structure',
    'Diag_Inner_Count',
    'Diag_Inner_Ratio',
    'Diag_Candidate',
    'Diag_Candidate_Hypothesis',
    'Diag_Candidate_Formula',
    'Diag_Revalidation_Z',
    'Diag_Revalidation_Score',
    'Diag_Revalidation_Pass',
    'Diag_Breakdown',
    'Diag_Max_Negative_Qty',
    'Diag_Suggested_Missing_Unit_Qty',
    'Diag_Suggested_Missing_Base_Qty',
    'Diag_Missing_History_Match',
    'Diag_Duplicate',
    'Diag_Evidence',
]

for col in diag_columns:
    if col in [
        'Diag_Is_Checked',
        'Diag_Revalidation_Pass',
        'Diag_Duplicate',
        'Diag_Missing_History_Match',
    ]:
        df[col] = False
    elif col in [
        'Diag_Root_Cause',
        'Diag_Confidence',
        'Diag_Tier_Structure',
        'Diag_Candidate_Hypothesis',
        'Diag_Candidate_Formula',
        'Diag_Breakdown',
        'Diag_Evidence',
    ]:
        df[col] = None
    else:
        df[col] = np.nan


# ============================================================
# 9.14 Apply Diagnosis
# ============================================================
anomaly_mask = (
    df['is_outlier_import']
    | df['is_missing_inbound_bill']
    | df['is_stock_out']
    | df['is_balance_tampered']
    | df['is_ghost_stock']
    | df['is_dead_last_item']
)

anomaly_indices = df.index[anomaly_mask]

if len(anomaly_indices) > 0:
    # ดึงเฉพาะ product_id ที่มีปัญหามาสร้างประวัติ เพื่อลดการใช้ Memory และเวลา
    anomaly_pids = df.loc[anomaly_indices, 'product_id'].unique()
    
    # สร้าง Dictionary เก็บประวัติของสินค้าแต่ละตัวไว้ล่วงหน้า (O(1) lookup time)
    # เฉพาะรายการที่มีปัญหาเท่านั้น เพื่อลดการ Scan ตารางใหม่ทุกรอบในลูป
    relevant_history = df[df['product_id'].isin(anomaly_pids)]
    history_dict = {pid: group for pid, group in relevant_history.groupby('product_id')}
    
    for idx in anomaly_indices:
        row = df.loc[idx]
        product_history = history_dict[row['product_id']].copy()
        diagnosis = diagnose_row(row, product_history)
        
        for col in diag_columns:
            df.at[idx, col] = diagnosis[col]


# ============================================================
# 10. Final Classification
# ============================================================
mask_has_diag = df['Diag_Root_Cause'].notna() & (df['Diag_Is_Checked'] == True)
df['Final_Diagnosis'] = np.where(mask_has_diag, df['Diag_Root_Cause'], df['Anomaly_Type'])


# ============================================================
# 11. Diagnostic Summary Columns
# ============================================================
df['Diag_Master_Unit'] = pd.to_numeric(
    df['Diag_Master_Unit'], errors='coerce'
)
df['Diag_Inner_Unit'] = pd.to_numeric(
    df['Diag_Inner_Unit'], errors='coerce'
)
df['Diag_Candidate'] = pd.to_numeric(
    df['Diag_Candidate'], errors='coerce'
)
df['Diag_Revalidation_Z'] = pd.to_numeric(
    df['Diag_Revalidation_Z'], errors='coerce'
)
df['Diag_Revalidation_Score'] = pd.to_numeric(
    df['Diag_Revalidation_Score'], errors='coerce'
)


# ============================================================
# 11.1 ★ Expected_Import: ใช้ Diag_Candidate แทนค่าเดิม ★
#      (แก้ไขจุดที่ 2 ส่วนที่ 2: Diagnosis overwrite)
#
#      ลำดับ Priority:
#      1. Diag_Candidate ที่ Re-validation ผ่าน → ใช้แทน
#      2. คง EWMA prediction ไว้ (จาก Section 3.1)
#      3. ถ้า import ปกติ → import เดิม
# ============================================================
diag_candidate_mask = (
    (df['Diag_Revalidation_Pass'] == True)
    & (df['Diag_Candidate'].notna())
    & (df['Diag_Candidate'] > 0)
)

df.loc[diag_candidate_mask, 'Expected_Import'] = (
    df.loc[diag_candidate_mask, 'Diag_Candidate']
)

# จำนวนสินค้าเป็นหน่วยเต็ม: กันไม่ให้ค่า EWMA ทศนิยมหลุดเป็นคำตอบสุดท้าย
qty_mask = (df['import'] > 0) & df['Expected_Import'].notna()
df.loc[qty_mask, 'Expected_Import'] = _round_positive_qty_series(
    df.loc[qty_mask, 'Expected_Import']
)

# อัปเดต Diff_Import ตาม Expected_Import ใหม่
df['Diff_Import'] = np.where(
    df['import'] > 0,
    df['import'] - df['Expected_Import'],
    np.nan
)

diff_mask = df['Diff_Import'].notna()
df.loc[diff_mask, 'Diff_Import'] = np.round(df.loc[diff_mask, 'Diff_Import'])


# ============================================================
# 12. ★ Focus_Columns ★
#     (แก้ไขจุดที่ 3: บอกว่าแต่ละ Anomaly ควรดูคอลัมน์ไหน)
# ============================================================
focus_conditions = [
    (df['is_outlier_import'] == True),
    (df['is_missing_inbound_bill'] == True),
    (df['is_ghost_stock'] == True),
    (df['is_stock_out'] == True),
    (df['is_balance_tampered'] == True),
    (df['is_dead_last_item'] == True),
]

focus_labels = [
    'import, Expected_Import, Diff_Import, Diag_Candidate, Diag_Breakdown',
    'suspected_missing_period, estimated_missing_qty, Diag_Suggested_Missing_Base_Qty',
    'days_idle, balances, adjusted_calc_balance',
    'balances, export, import',
    'balances, adjusted_calc_balance, net_flow',
    'days_idle, balances, median_inter_sale_days',
]

df['Focus_Columns'] = np.select(
    focus_conditions,
    focus_labels,
    default=''
)


# ============================================================
# 13. ★ Anomaly_Evidence ★
#     (แก้ไขจุดที่ 4: บิล/ลืมคีย์ต้องมีเหตุผลรับรอง)
#
#     เชื่อมโยง Diag_Evidence จาก Diagnosis layer
#     กลับมายัง Detection layer เพื่อให้ทุก anomaly มีเหตุผล
# ============================================================
df['Anomaly_Evidence'] = np.where(
    df['Diag_Is_Checked'] == True,
    df['Diag_Evidence'].fillna('Diagnosis completed — no specific evidence'),
    np.where(
        df['Anomaly_Type'] != '⚪ ปกติ (Normal)',
        'Detection flag only — pending diagnosis',
        ''
    )
)


# ============================================================
# 13.1 Trace Back To Suspected Bill
# ============================================================
trace_mask = (
    (df['Diag_Is_Checked'] == True)
    | (df['Anomaly_Type'] != '⚪ ปกติ (Normal)')
)

df['Trace_Back_Date'] = np.where(
    trace_mask,
    df['DATE'].dt.strftime('%Y-%m-%d'),
    ''
)

df['Trace_Back_Bill'] = np.where(trace_mask, df['Bill'], '')

df['Trace_Hypothesis'] = np.where(
    df['Diag_Candidate_Hypothesis'].notna(),
    df['Diag_Candidate_Hypothesis'],
    np.where(trace_mask, df['Final_Diagnosis'], None)
)

df['Trace_Formula'] = np.where(
    df['Diag_Candidate_Formula'].notna(),
    df['Diag_Candidate_Formula'],
    np.where(trace_mask, df['Anomaly_Evidence'], None)
)

df['Trace_Candidate_Import'] = np.where(
    df['Diag_Candidate'].notna(),
    df['Diag_Candidate'],
    np.where(
        trace_mask & df['Expected_Import'].notna(),
        df['Expected_Import'],
        np.nan
    )
)


# ============================================================
# 13.2 Quantity Output Types
# ============================================================
quantity_output_cols = [
    'Expected_Import',
    'Diff_Import',
    'estimated_missing_qty',
    'adjusted_import',
    'adjusted_calc_balance',
    'net_flow',
    'calc_balance_before_row',
    'period_end_balance',
    'inferred_import',
    'zero_stock_expected_import',
    'zero_stock_simulated_balance',
    'Diag_Candidate',
    'Diag_Max_Negative_Qty',
    'Diag_Suggested_Missing_Unit_Qty',
    'Diag_Suggested_Missing_Base_Qty',
    'Trace_Candidate_Import',
]

for qty_col in quantity_output_cols:
    if qty_col in df.columns:
        df[qty_col] = (
            pd.to_numeric(df[qty_col], errors='coerce')
            .round()
            .astype('Int64')
        )


# ============================================================
# 14. Result
# ============================================================
# df คือ DataFrame ผลลัพธ์สุดท้าย
#
# คอลัมน์เดิมยังอยู่ครบ เช่น:
# Expected_Import        ← ★ ใช้ EWMA + Diag_Candidate (ไม่ใช่ median)
# Diff_Import
# ZScore_Import          ← ★ คำนวณจากเฉพาะ Bill IB/IBK/DM
# is_outlier_import      ← ★ กรอง valid bill ก่อน
# is_valid_import_bill   ← ★ flag ว่า bill ขึ้นต้นด้วย IB/IBK/DM
# is_suspected_ghost
# is_dead_last_item
# is_missing_inbound_bill
# estimated_missing_qty
# adjusted_import
# adjusted_calc_balance
# is_balance_tampered
# is_stock_out
# Anomaly_Type
#
# และเพิ่ม Diagnosis:
# Diag_Master_Unit
# Diag_Inner_Unit
# Diag_Base_Unit
# Diag_Tier_Structure
# Diag_Candidate
# Diag_Candidate_Hypothesis
# Diag_Revalidation_Z
# Diag_Revalidation_Pass
# Diag_Breakdown
# Diag_Max_Negative_Qty
# Diag_Suggested_Missing_Base_Qty
# Diag_Missing_History_Match
# Diag_Duplicate
# Diag_Root_Cause
# Diag_Confidence
# Diag_Evidence
# Final_Diagnosis
#
# ★ คอลัมน์ใหม่:
# Focus_Columns          ← บอกคอลัมน์ที่ต้องดูตาม Anomaly
# Anomaly_Evidence       ← เหตุผลรับรองจาก Diagnosis
# Trace_Back_Date        ← วันที่ของแถว/บิลที่ต้องย้อนกลับไปตรวจ
# Trace_Back_Bill        ← เลขบิลที่ต้องย้อนกลับไปตรวจ
# Trace_Hypothesis       ← สมมติฐานที่ใช้ชี้บิลผิด
# Trace_Formula          ← สูตร/เหตุผลที่ทำให้ย้อนกลับได้
# Trace_Candidate_Import ← ยอดรับเข้าที่ควรเป็นหลังย้อนกลับ
#
# ============================================================
# 15. เลือกเฉพาะคอลัมน์ที่จำเป็น (Filter Necessary Columns)
# ============================================================
necessary_columns = [
    'DATE',
    'product_id',
    'Bill',
    'details',
    'import',
    'export',
    'balances',
    'Expected_Import',
    'Diff_Import',
    'Trace_Back_Date',
    'Trace_Back_Bill',
    'Trace_Hypothesis',
    'Trace_Formula',
    'Trace_Candidate_Import',
    'ZScore_Import',
    'is_outlier_import',
    'is_missing_inbound_bill',
    'is_ghost_stock',
    'is_dead_last_item',
    'is_stock_out',
    'is_balance_tampered',
    'suspected_missing_period',
    'estimated_missing_qty',
    'adjusted_import',
    'adjusted_calc_balance',
    'net_flow',
    'calc_balance_before_row',
    'period_end_balance',
    'inferred_import',
    'zero_stock_expected_import',
    'Anomaly_Type',
    'Final_Diagnosis',
    'Diag_Root_Cause',
    'Diag_Candidate',
    'Diag_Candidate_Hypothesis',
    'Diag_Candidate_Formula',
    'Diag_Revalidation_Z',
    'Diag_Revalidation_Pass',
    'Diag_Breakdown',
    'Diag_Max_Negative_Qty',
    'Diag_Suggested_Missing_Base_Qty',
    'Diag_Missing_History_Match',
    'Diag_Duplicate',
    'Anomaly_Evidence',
    'Focus_Columns'
]
df = df[[c for c in necessary_columns if c in df.columns]]

# ============================================================
# จบ Pipeline
# ============================================================
