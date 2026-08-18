## 🚀 MASTER LOGIC & PIPELINE SPECIFICATION (CAVEMAN MODE)

### 1. DATA SCOPE RULES
* **Strict Filter (Use ONLY `DMIB`, `IBK`, `IBDM`):**
  * Robust Z-Score calculation
  * Inner Bar extraction
  * Basic Bar detection
* **Full Data (NO Filtering):**
  * Expected Import calculation & Reverse operations (Must use 100% of raw data to prevent distortion).

### 2. INNER BAR ALGORITHM
* **Group:** `product_id` & `unit`
* **Math:** Calculate Square Root ($\sqrt{\text{Total}}$) of aggregated volume.
* **Ceiling Rule:** 
  * If integer $\rightarrow$ Use exact value.
  * If decimal (e.g., $12.x$) $\rightarrow$ **Ceiling / Round UP immediately** (e.g., $12.x \rightarrow 13$). No rounding based on 0.5 rules.

### 3. BASIC BAR ALGORITHM (Performance Optimized)
* **Target:** Detect fractional/split imports (unit pieces smaller than Inner Bar).
* **Vectorized Modulo Check:** 
  * Formula: `remainder = quantity % inner_bar`
  * Condition: If `remainder > 0` anywhere in the filtered dataset for a product $\rightarrow$ **Stop & Flag:** `basic_bar = 1`.
* **Prohibition:** NO row-by-row `for loop` to prevent latency on large datasets ($\sim$200k rows). Use Pandas vectorized methods.

### 4. EXPECTED IMPORT RULE
* **Strict Integer Constraint:** Expected Import must **NEVER be a decimal** (e.g., no 5.x).
* **Method:** Driven strictly by core hypotheses matched to predefined unit tiers (Master Pack, Inner Bar, Basic Bar) to output exact whole numbers.

### 5. ANOMALY, OUTLIER & KEYING ERROR LOGIC
* **Dead / Stagnant Stock Detection:** 
  * If stock sits unusually long (exceeds $2\times$ average sales cycle) without movement, and suddenly clears out immediately upon a new import arrival $\rightarrow$ Flag legacy stock as inactive/dead.
* **Typo / Unit Mix-up & Backtracking:**
  * When an Outlier occurs (e.g., spike to 24 units vs. normal 1-3 range), check if subtraction/division aligns with Inner/Basic bars (e.g., identifying cross-keyed master vs. basic units).
* **Split-Bill / Missing-Entry Re-check:**
  * If an anomaly value appears (e.g., 6 units), test splitting hypotheses (e.g., dividing into 3 + 3). 
  * If the split values pass the outlier test without triggering violations, flag as combined/missing-entry records rather than true outliers.
* **Sensitivity Tuning:** 
  * Keep outlier detection filters **less sensitive** (wider variance tolerance) to prevent valid irregular volumes from being aggressively purged.